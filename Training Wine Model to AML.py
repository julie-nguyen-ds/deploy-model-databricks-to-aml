# Databricks notebook source
# MAGIC %md # Deploy Models from Databricks to Azure ML Endpoint
# MAGIC This example notebook demonstrates how to deploy models trained in Databricks to Azure ML Managed Endpoint on Azure Machine Learning. The main goal is to be able to deploy using code only, and in a seamless manner, with only a Databricks notebook and the right configurations.
# MAGIC This notebook is the part one of the series [How to deploy model trained on Databricks to Azure ML Endpoint or AKS](https://jnguyends.medium.com/in-depth-guide-deploy-models-from-databricks-to-azure-ml-2023-6d71572eb6f7). 
# MAGIC
# MAGIC **Notebook Cluster Config:** DBR 13.0 ML / Standard DS3_v2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Databricks to AML
# MAGIC **Important:** In order to successfully let Databricks communicate with Azure ML, you will first need to **grant access** to Databricks to write and read from Azure ML. You can find how-to in the first section of the guide here: [Setup Managed Identity roles for Access Permission](https://jnguyends.medium.com/in-depth-guide-deploy-models-from-databricks-to-azure-ml-2023-6d71572eb6f7). 
# MAGIC
# MAGIC When that's done, you can move on to the next steps.

# COMMAND ----------

# MAGIC %md ### Install Azure Machine Learning Dependencies
# MAGIC Now, you can install the following Python packages which contain integration code of AzureML with MLflow, and will help create endpoints and deploy your model. You can either use `pip install` or [install them directly on your cluster](https://learn.microsoft.com/en-us/azure/databricks/libraries/cluster-libraries#--install-a-library-on-a-cluster):
# MAGIC
# MAGIC - `azureml-mlflow`
# MAGIC - `azure-ai-ml`

# COMMAND ----------

pip install azure-ai-ml

# COMMAND ----------

pip install azureml-mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect MLFlow to Azure ML Server
# MAGIC After that, configure your resources information to retrieve your Azure ML workspace like in the code snipped below.

# COMMAND ----------

import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# TODO: Enter details of your Azure Machine Learning workspace
subscription_id = "<Subscription ID of your resource group>"
resource_group = "<Resource group having your resources>"
workspace_name = "<Your azure workspace name>"

# Retrieves your Azure ML resources with already set up Managed Identity
ml_client = MLClient(credential=DefaultAzureCredential(),
                        subscription_id=subscription_id, 
                        workspace_name=workspace_name,
                        resource_group_name=resource_group)

# Retrieves MLflow tracking URI of Azure ML workspace
aml_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

# Changes MLflow tracking URI to Azure ML server
mlflow.set_tracking_uri(aml_tracking_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and Log Model to AML
# MAGIC Now, we want to train a simple model and register it to Azure ML Model Registry. We are using the wine quality dataset to create a wine quality scoring model.

# COMMAND ----------

# MAGIC %md ### Load Wine Quality Dataset
# MAGIC The Dataset used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality. By P.Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn

data = pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv", sep=";")

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# COMMAND ----------

# MAGIC %md ### Train and Track ML Model Experiments
# MAGIC To track MLflow experiments on Azure ML, you need to create an MLflow experiment and set the experiment. Else using MLflow will return the exception `BadRequest: Experiment ID must be a GUID.`

# COMMAND ----------

# Creates and sets the experiment when using MLflow with Azure ML
experiment_name = "wine_quality_experiment"
mlflow.set_experiment(experiment_name=experiment_name)

# COMMAND ----------

alpha = 0.5
l1_ratio = 0.5
artifact_path = "model"

with mlflow.start_run() as run:
    # Keep the metadata of the run
    run_id = run.info.run_id
    
    # Train your model
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})

    # Infer model signature
    signature = mlflow.models.infer_signature(model_input=test_x[:10], model_output=predicted_qualities[:10])

    # Log the model to the experiment
    mlflow.sklearn.log_model(lr, artifact_path, signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Model on Azure ML Model Registry
# MAGIC Once you are satisfied with your model experimentation, you can register your best model version by using `register_model`.

# COMMAND ----------

registered_model_name = "wine_quality"
registered_model = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", registered_model_name)

# COMMAND ----------

# MAGIC %md ### Load Model for Batch Predictions
# MAGIC
# MAGIC You can load a model version by specifying its name and version number. Below, we load the latest version of our registered model using [mlflow.pyfunc.load_model](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model) and apply it for inference on our test dataset

# COMMAND ----------

import mlflow.pyfunc

model_version = registered_model.version
model_version_uri = f"models:/{registered_model_name}/{model_version}"
model_version = mlflow.pyfunc.load_model(model_version_uri)
model_version.predict(test_x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve the model with AML Endpoint
# MAGIC Azure ML Endpoint are off-the-shelf solution to deployment where we don't have access to the underlying infrastructure. They are fast to deploy, require minimal configuration and maintenance whilst being less customizable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Azure ML Endpoint
# MAGIC In order to create an Azure ML Managed Endpoint, we need as input a configuration file in json format with a few parameters.
# MAGIC - `auth_mode`: Determines authentication mode for the endpoint. Can be `"key"`, `"anonymous"` or `"aad"`.
# MAGIC - `identity / type`: Specifies the type of identity assigned to the endpoint. Can be `"none"`, `"system_assigned"` or `"user_assigned"`.

# COMMAND ----------

import json

# Write the endpoint configuration file 
endpoint_config_path = "endpoint_config.json"
endpoint_config = {
    "auth_mode": "key",
    "identity": {
        "type": "system_assigned"
    }
}
with open(endpoint_config_path, "w") as outfile:
    outfile.write(json.dumps(endpoint_config))

# COMMAND ----------

# MAGIC %md
# MAGIC Since we want to deploy on Azure ML, we retrieve the deployment_client associated to our workspace using Azure ML tracking URI. Then, we create the endpoint using the configuration file created above and give it a name.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# Create the deployment client linked to Azure ML workspace
deployment_client = get_deploy_client(aml_tracking_uri)  

# Create a AML managed endpoint 
endpoint_name = "wine-endpoint-test"
endpoint = deployment_client.create_endpoint(
    name=endpoint_name,
    config={"endpoint-config-file": endpoint_config_path}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the Model on the Endpoint and Assign Traffic
# MAGIC When endpoint are created, they are initially empty and waiting for deployment to be made on it. An endpoint can host multiple deployments.
# MAGIC Let's create a first deployment. First, we specify the compute resources we want to allocate to that deployment in a configuration file.

# COMMAND ----------

blue_deployment_name = "default"
deploy_config = {
    "instance_type": "Standard_DS2_v2",
    "instance_count": 1,
}
deployment_config_path = "deployment_config.json"
with open(deployment_config_path, "w") as outfile:
    outfile.write(json.dumps(deploy_config))

# COMMAND ----------

# MAGIC %md
# MAGIC Then, we can use the `deployment_client` to create a deployment on the endpoint we created. Inputs are our model name and version we logged in the Model Registry and the configuration file we've just created.

# COMMAND ----------

model_name = registered_model_name
version = registered_model.version

blue_deployment = deployment_client.create_deployment(
    name=blue_deployment_name,
    endpoint=endpoint_name,
    model_uri=f"models:/{model_name}/{version}",
    config={"deploy-config-file": deployment_config_path},
)    

# COMMAND ----------

# MAGIC %md 
# MAGIC The deployment will take a few minutes to roll out. Before we can call our model through endpoint requests, we need to update the traffic percent our deployment gets by the endpoint.

# COMMAND ----------

traffic_config = {"traffic": {deployment_name: 100}}
traffic_config_path = "traffic_config.json"
with open(traffic_config_path, "w") as outfile:
    outfile.write(json.dumps(traffic_config))

deployment_client.update_endpoint(
    endpoint=endpoint_name,
    config={"endpoint-config-file": traffic_config_path},
)  

scoring_uri = deployment_client.get_endpoint(endpoint=endpoint_name)["properties"]["scoringUri"]
print(scoring_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC After that, you are done and you can test out your served model! 
# MAGIC
# MAGIC You can either use Azure ML UI or make requests on Postman / Databricks. Here's a request example below.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Query AML Endpoint with requests

# COMMAND ----------

import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
data =  {
  "input_data": {
    "columns": [
      "fixed acidity",
      "volatile acidity",
      "citric acid",
      "residual sugar",
      "chlorides",
      "free sulfur dioxide",
      "total sulfur dioxide",
      "density",
      "pH",
      "sulphates",
      "alcohol"
    ],
    "index": [[111]],
    "data": [[8.4, 0.620, 0.09, 2.20, 0.084, 11.0, 108.0, 0.99640, 3.15, 0.66, 9.8]]
  }
}

body = str.encode(json.dumps(data))
url = scoring_uri

# TODO: Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = '<your endpoint API key>'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'default' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))


# COMMAND ----------


