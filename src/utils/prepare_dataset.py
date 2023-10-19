# src/utils/prepare_dataset.py

from azureml.core import Dataset
from azureml.core import Workspace
import os

# Connect to workspace
subscription_id = '4914df7c-27f4-4064-af6c-de1c82735e9b'
resource_group = 'HARVEY-resources'
workspace_name = 'Harvey_AML'
workspace = Workspace(subscription_id, resource_group, workspace_name)

# Get the datastore
datastore_name = 'harvey_blob_datastore'
datastore = Datastore.get(workspace, datastore_name)

# Create a file dataset
dataset_name = 'plant_disease_images'
path_on_datastore = './train'  # Assuming your blob has 'train' directory with images
datastore_paths = [(datastore, path_on_datastore)]
dataset = Dataset.File.from_files(path=datastore_paths)

# Register the dataset
dataset.register(workspace=workspace,
                 name=dataset_name,
                 description='Plant Disease Image Dataset',
                 create_new_version=True)

print(f"Dataset '{dataset_name}' registered.")
