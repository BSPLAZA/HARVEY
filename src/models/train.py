import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from load_model import load_resnet50_model
from tqdm import tqdm
import argparse
import sys

# Suppress annoying warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a modified ResNet-50 model.")

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
    parser.add_argument("--drop_last", type=bool, default=True,
                        help="Drop the last batch if its size is inconsistent.")

    args = parser.parse_args()
    return args


if torch.cuda.is_available():
    AUTOCAST_EN = True
    torch.backends.cudnn.benchmark = True
else:
    AUTOCAST_EN = False


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

    train_data = ImageFolder(root=args.train_data_path,
                             transform=train_transform)
    valid_data = ImageFolder(root=args.valid_data_path,
                             transform=valid_transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=args.drop_last)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=args.drop_last)

    return train_loader, valid_loader


def train(model, device, train_loader, criterion, optimizer, scaler, epoch):
    model.train()
    train_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        for param in model.parameters():
            param.grad = None
        with autocast(dtype=torch.bfloat16, enabled=AUTOCAST_EN):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        pbar.set_postfix(loss="{:.5f}".format(loss.item()))
    avg_loss = train_loss / len(train_loader)
    print('\033[96m' + f"Average Loss = {avg_loss:.5f}" + '\033[0m')
    return avg_loss


def validate(model, device, valid_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            with autocast(dtype=torch.bfloat16, enabled=AUTOCAST_EN):
                output = model(data)
                loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss / len(valid_loader)


def main(args):
    # Load model and set device
    model = load_resnet50_model(args.model_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device under use:", torch.cuda.get_device_name(), "(GPU)")
    else:
        device = torch.device("cpu")
        print("Device under use: CPU")
    model.to(device)

    # Load data
    train_loader, valid_loader = load_data(args)

    # Define the Loss Function, Optimizer, and Scaler for mixed precision
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Start TensorBoard Summary Writer
    writer = SummaryWriter()

    # Training loop with Early Stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, 100 + 1):
        try:
            train_loss = train(model, device, train_loader,
                               criterion, optimizer, scaler, epoch)
            val_loss = validate(model, device, valid_loader, criterion)

            # Log training and validation loss to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validate', val_loss, epoch)

            # Print validation loss
            print('\033[91m' + f"Validation Loss = {val_loss:.5f}" + '\033[0m')

            # Check if the validation loss is a new best
            if val_loss < best_val_loss:
                print("New best model found!")
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping: stop training if validation loss hasn't improved for 10 epochs
            if epochs_without_improvement == 10:
                print(
                    "No improvement in validation loss for 10 epochs. Stopping training.")
                break

        except KeyboardInterrupt:
            print("Interrupted by user. Stopping training early.")
            break

    # Save the final model
    torch.save(model.state_dict(), args.save_path)

    # Close TensorBoard Summary Writer
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
