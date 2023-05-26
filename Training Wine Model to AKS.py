# Databricks notebook source
# MAGIC %md # Deploy Models from Databricks to AKS
# MAGIC This example notebook demonstrates how to deploy models trained in Databricks to Azure Kubernetes Compute on Azure Machine Learning. The main goal is to be able to deploy using code only, and in a seamless manner, with only a Databricks notebook and the right configurations.
# MAGIC This notebook is the part two of the series [How to deploy model trained on Databricks to Azure ML Endpoint or AKS](https://jnguyends.medium.com/in-depth-guide-deploy-models-from-databricks-to-azure-ml-2023-6d71572eb6f7). 
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
# MAGIC Now, you can install the following Python packages which contain integration code of AzureML with MLflow, and will help create AKS deployment resources. You can either use `pip install` or [install them directly on your cluster](https://learn.microsoft.com/en-us/azure/databricks/libraries/cluster-libraries#--install-a-library-on-a-cluster):
# MAGIC
# MAGIC - `azureml-mlflow`
# MAGIC - `azure-ai-ml`
# MAGIC - `azureml-core`

# COMMAND ----------

pip install azure-ai-ml

# COMMAND ----------

pip install azureml-mlflow

# COMMAND ----------

pip install azureml-core

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
experiment_name = "wine_quality_aks"
mlflow.set_experiment(experiment_name=experiment_name)

# COMMAND ----------

# MAGIC %md After that, we train a model and log the model experiment to MLflow. 
# MAGIC
# MAGIC Notice how we associate a signature to our model. A model signature in MLflow defines the schema of a model’s inputs and outputs. It is not mandatory but it’s good practice to log model with their signature. Azure Machine Learning enforces compliance with it, both in terms of the number of inputs and their types when using online inference endpoint.

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
# MAGIC ### Register the Model on AML Model Registry
# MAGIC Once you are satisfied with your model experimentation, you can register your best model version by using `register_model`.

# COMMAND ----------

registered_model_name = "wine_quality"
registered_model = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", registered_model_name)

# COMMAND ----------

# MAGIC %md ### Test Model for Batch Predictions
# MAGIC
# MAGIC You can load a model version by specifying its name and version number. Below, we load the latest version of our registered model using [mlflow.pyfunc.load_model](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model) and apply it for inference on our test dataset.

# COMMAND ----------

import mlflow.pyfunc

model_version = registered_model.version
model_version_uri = f"models:/{registered_model_name}/{model_version}"
model_version = mlflow.pyfunc.load_model(model_version_uri)
model_version.predict(test_x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve the model with AKS Compute Endpoint
# MAGIC When you create an AKSCompute in Azure ML, it is provisioned as a managed service specifically designed for running machine learning workloads. 
# MAGIC Azure ML abstracts away the underlying infrastructure management, making it easier to deploy and manage machine learning models. 

# COMMAND ----------

from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core import Workspace
from azureml.core.authentication import MsiAuthentication

# Retrieves your Azure ML resources with already set up Managed Identity
ws = Workspace(subscription_id=subscription_id,
               resource_group=resource_group,
               workspace_name=workspace_name,
               auth=MsiAuthentication())

print("Found workspace {} at location {}".format(ws.name, ws.location))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Create AKS Compute Endpoint
# MAGIC You can either deploy your model endpoint to a new AKS resources, or you can attach the model endpoint to an existing AKS resource. We will demonstrate both way. Let's start with a new AKS resource creation, managed by Azure ML.

# COMMAND ----------

# MAGIC %md #### a. Attach Endpoint to New AKS resource
# MAGIC If you want to **use an existing AKS resource, go to step b.**
# MAGIC
# MAGIC Here, we create our AKSCompute Endpoint which takes about 5 minutes to roll out. The AKSCompute endpoint name `aks_name` has some restriction. It must start with a letter, end with a letter or digit, and be between 2 and 16 characters in length. It can include letters, digits and dashes.

# COMMAND ----------


# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration(vm_size = "Standard_DS2_v2",
                                                    agent_count = 3,
                                                    location = "westus")

# Create the cluster
aks_endpoint_name = "wine-endpoint-1"
aks_target = ComputeTarget.create(workspace=ws, 
                                  name=aks_name, 
                                  provisioning_configuration=prov_config)

aks_target.wait_for_completion(show_output = True)

# COMMAND ----------

# MAGIC %md #### b. Attach Endpoint to Existing AKS resource
# MAGIC You might need to grant proper permissions for Databricks to attach the AKS resource to Azure ML. 
# MAGIC
# MAGIC You can follow the same steps as we did above to grant Databricks permission to AKS. If you encounter errors, know that there is a minimum pool node size requirements in order to deploy endpoints on an existing AKS resource. 

# COMMAND ----------

# TODO: Fill your AKS resource information
existing_aks_name = "<existing aks resource name>"
subscription_id = "<azure subscription id>" 
resource_group = "<resource group of aks>"
aks_resource_id = f"/subscriptions/{subscription_id}/resourcegroups/{resource_group}/providers/Microsoft.ContainerService/managedClusters/{existing_aks_name}"

# Create configuration and attach the AKS pool to an AKSCompute endpoint
aks_endpoint_name = "wine-endpoint-1"
existing_attach_config = AksCompute.attach_configuration(resource_id=aks_resource_id)
aks_target = ComputeTarget.attach(ws, aks_endpoint_name, existing_attach_config)

aks_target.wait_for_completion(show_output = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Create Model Deployments on Endpoint
# MAGIC Finally, we can create our model deployment on the endpoint. Let's create the json configuration file for the deployment.

# COMMAND ----------

import json
deployment_config = {"computeType": "aks", "computeTargetName": aks_endpoint_name}

deployment_config_path = "deployment_config.json"
with open(deployment_config_path, "w") as outfile:
    outfile.write(json.dumps(deployment_config))

# COMMAND ----------

# MAGIC %md
# MAGIC We can then deploy our model from Model Registry using the deployment_client.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# set the tracking uri as the deployment client
client = get_deploy_client(mlflow.get_tracking_uri())
version = registered_model.version

# set the deployment config
deploy_path = "deployment_config.json"
test_config = {'deploy-config-file': deploy_path}

# define the model path and the name is the service name
# the model gets registered automatically and a name is autogenerated using the "name" parameter below 
deployment_name = "default"
client.create_deployment(model_uri=f"models:/{registered_model_name}/{version}",
                         config=test_config,
                         name=deployment_name)

# Fetch the endpoint scoring uri
scoring_uri = client.get_endpoint(endpoint=aks_endpoint_name)["properties"]["scoringUri"]
print(scoring_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC The deployment will take about 5 minutes to roll out. After that, you are done and you can test out your served model on AKS! This will also create a AKS cluster in your resource group which you can configure and govern as you need.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Test and Query AKS Compute Endpoint with requests
# MAGIC Here's how to do a request on the model deployed endpoint you just created.
# MAGIC
# MAGIC **TODO:** Replace `api_key` with the primary/secondary key or AMLToken for the endpoint.

# COMMAND ----------

## Use it
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


