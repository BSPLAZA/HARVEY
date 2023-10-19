# src/models/ObjectDetectionTraining.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from azureml.core import Workspace, Dataset
from azureml.core import Model
import os

# Define constants
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 40  # Total unique classes

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
subscription_id = '4914df7c-27f4-4064-af6c-de1c82735e9b'
resource_group = 'HARVEY-resources'
workspace_name = 'Harvey_AML'
workspace = Workspace(subscription_id, resource_group, workspace_name)
dataset_name = 'plant_disease_images'
dataset = Dataset.get_by_name(workspace, name=dataset_name)

# Define transformations and data loader
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset.mount(), x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'valid']}

# Load a pre-trained model and modify the last layer
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Train the model
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss / len(dataloaders['train'].dataset)}")

# Save the model
torch.save(model.state_dict(), './data/processed/plant_disease_model.pth')
print("Model saved.")

model = Model.register(model_path="./data/processed/plant_disease_model.pth",
                       model_name="plant_disease_detection",
                       tags={'type': "classification", 'framework': "pytorch"},
                       description="Plant disease detection model",
                       workspace=workspace)