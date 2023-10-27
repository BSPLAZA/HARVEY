import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from load_model import load_resnet50_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a modified ResNet-50 model.")
    
    # Paths
    parser.add_argument("--model_path", default="../../models/ResNet50-Plant-model-80.pth", 
                        help="Path to the initial model checkpoint.")
    parser.add_argument("--train_data_path", default="../../.kaggle/train", 
                        help="Path to the training dataset.")
    parser.add_argument("--valid_data_path", default="../../.kaggle/valid", 
                        help="Path to the validation dataset.")
    parser.add_argument("--save_path", default="../../models/ResNet50-Plant-model-Final.pth", 
                        help="Path to save the final trained model.")
    parser.add_argument("--checkpoint_path", default="../../models/checkpoint_best.pth", 
                        help="Path to save the best model checkpoint based on validation loss.")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Initial learning rate.")
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Number of epochs to train.")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Number of epochs without improvement in validation loss to wait before stopping training.")
    
    args = parser.parse_args()
    return args

def load_data(args):
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

    train_data = ImageFolder(root=args.train_data_path, transform=train_transform)
    valid_data = ImageFolder(root=args.valid_data_path, transform=valid_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, valid_loader

def train(epoch, model, train_loader, criterion, optimizer, device):
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

def validate(model, valid_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss / len(valid_loader)

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load model and move to device
    model = load_resnet50_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device under use: " + torch.cuda.get_device_name())
    model.to(device)
    
    # Load data
    train_loader, valid_loader = load_data(args)
    
    # Define the Loss Function, Optimizer, and Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    
    # TensorBoard Summary Writer
    writer = SummaryWriter()
    
    # Training loop with Early Stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(epoch, model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, valid_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)

        # Logging to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print("Stopping early!")
            break

    writer.close()
    
    # Save the model
    torch.save(model, args.save_path)

if __name__ == "__main__":
    main()
