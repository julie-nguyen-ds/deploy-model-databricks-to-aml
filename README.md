## In-Depth Guide: Deploy Models from Databricks to Azure ML (2023)

![databricks-azureml.png](resources%2Fdatabricks-azureml.png)

Follow this **[step-by-step guide]([https://jnguyends.medium.com/in-depth-guide-deploy-models-from-databricks-to-azure-ml-2023-6d71572eb6f7](https://jnguyends.medium.com/in-depth-guide-deploy-models-from-databricks-to-azure-ml-2023-6d71572eb6f7?source=friends_link&sk=aee6918878e18b416763fc91ebcf8b9e))** to run the repository.

In this repository, we explore two ways you can easily and seamlessly deploy models trained on Databricks to Azure ML endpoints using Databricks notebooks.
These notebooks are meant to be run entirely on Azure Databricks for end to end deployment. There is no need to switch to yaml files or Azure ML scripts.

We focus on deploying our ML models on two main compute types:
- Azure ML Managed Endpoint ([notebook](https://github.com/julie-nguyen-ds/deploy-model-databricks-to-aml/blob/main/Training%20Wine%20Model%20to%20AML.ipynb))
- Azure AKS Compute Endpoint v.1 ([notebook](https://github.com/julie-nguyen-ds/deploy-model-databricks-to-aml/blob/main/Training%20Wine%20Model%20to%20AKS.ipynb))

### Architecture Flow
![databricks-aml-deployment.gif](resources%2Fdatabricks-aml-deployment.gif)

### Technical Stack
- Azure Databricks
- Azure Machine Learning
- Azure Kubernetes Services
- Azure Managed Identity
- MLFlow

### Additional Links
- Official documentation on [Managed Endpoint vs Kubernetes Endpoint](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2#managed-online-endpoints-vs-kubernetes-online-endpoints)
- Official documentation on [AKSCompute (v1) vs KubernetesCompute (v2)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-anywhere?view=azureml-api-2#kubernetescompute-and-legacy-akscompute)
