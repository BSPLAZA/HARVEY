import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from load_model import load_resnet50_model


# Load the modified model
model = load_resnet50_model('../../models/ResNet50-Plant-model-80.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dataset paths
TRAIN_DATA_PATH = '../../.kaggle/train'
VALID_DATA_PATH = '../../.kaggle/valid'

# Preprocess the dataset with Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
valid_data = ImageFolder(root=VALID_DATA_PATH, transform=valid_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

# Define the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# TensorBoard Summary Writer
writer = SummaryWriter()


def train(epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    return train_loss / len(train_loader)


def validate():
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss / len(valid_loader)


# Training loop with Early Stopping
NUM_EPOCHS = 100
patience = 10
epochs_without_improvement = 0
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train(epoch)
    val_loss = validate()

    # Logging to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

    # Checkpoint and Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), '../../models/checkpoint_best.pth')
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print("Stopping early!")
        break

writer.close()

# Save the model
model_save_path = '../../models/ResNet50-Plant-model-Final.pth'
torch.save(model, model_save_path)
