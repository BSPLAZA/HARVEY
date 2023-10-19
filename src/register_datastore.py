# src/utils/register_datastore.py

from azureml.core import Workspace, Datastore

# Replace these with your Azure ML Workspace details
subscription_id = '4914df7c-27f4-4064-af6c-de1c82735e9b'
resource_group = 'HARVEY-resources'
workspace_name = 'Harvey_AML'

# Get the Azure ML workspace
workspace = Workspace(subscription_id, resource_group, workspace_name)

# Define the blob datastore parameters
account_name = 'harvey2023storage'  
account_key = 'Nm3s2CBl8eMir1K6XCS+aQiNSdiDLDRYuiXf7qWEkzZH44H/Vs6xIQp9TbNBOACwqzrwKX4BbNiY+AStU9MDww=='
container_name = 'harvey'      
datastore_name = 'harvey_blob_datastore'  # Name you want to give your datastore in Azure ML

# Register the datastore
blob_datastore = Datastore.register_azure_blob_container(
    workspace=workspace, 
    datastore_name=datastore_name, 
    account_name=account_name,
    container_name=container_name, 
    account_key=account_key
)

print(f"Datastore '{datastore_name}' registered.")