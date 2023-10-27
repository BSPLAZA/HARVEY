import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Run by executing command, 'tensorboard --logdir=runs'
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from load_model import load_resnet50_model
from tqdm import tqdm

# Suppress annoying warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning) 

# File path
MODEL_OPEN_PATH = '../../models/ResNet50-Plant-model-80.pth'
MODEL_SAVE_PATH = '../../models/ResNet50-Plant-model-Final.pth'
TRAIN_DATA_PATH = '../../.kaggle/train'
VALID_DATA_PATH = '../../.kaggle/valid'

# Train parameter
if torch.cuda.is_available():
    NUM_EPOCHS = 100                        # Number of max epochs
    PATIENCE = 10                           # Used for determining when to stop training
    BATCH_SIZE = 32                         # Number of data processed at once
    NUM_WORKERS = 4                         # Tells the data loader instance how many sub-processes to use for data loading
    PIN_MEMORY = True                       # Automatically put fetched data Tensors in pinned memory, enabling faster data transfer to CUDA-enabled GPUs
    AUTOCAST_EN = True                      # Enables mixed precision computations for CUDA-enabled GPUs
    DROP_LAST = True                        # Required when running autocast
    torch.backends.cudnn.benchmark = True   # Tells CUDA framework to auto-tune the best algorithm to use for your hardware
else:
    NUM_EPOCHS = 25
    PATIENCE = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    PIN_MEMORY = False
    AUTOCAST_EN = False
    DROP_LAST = False
    

def load_data():
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

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,  drop_last = DROP_LAST)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,  drop_last = DROP_LAST)
    
    return train_loader, valid_loader


def train(model, device, train_loader, criterion, optimizer, scaler, epoch):
    model.train()
    train_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")  
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        for param in model.parameters(): # Zeroing out the gradients by setting the grad attribute to None solves performance overhead issues
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


def main():   
    
    # Load model, set device, and print hardware info
    model = load_resnet50_model(MODEL_OPEN_PATH)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device under use: \033[92m" + torch.cuda.get_device_name() + '\033[0m' + " (GPU)")
    else:
        import platform
        device = torch.device("cpu")
        print("Device under use: \033[94m" + platform.processor() + '\033[0m' + " (CPU)")
    model.to(device)
    
    # Load dataset
    train_loader, valid_loader = load_data()
    
    # Define the Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Start TensorBoard Summary Writer
    writer = SummaryWriter()
        
    # Training loop with Early Stopping
    epochs_without_improvement = 0
    best_val_loss = float('inf')

    print('\033[93m' + "Press CTRL+C to exit at any time..." + '\033[0m')  

    for epoch in range(1, NUM_EPOCHS + 1):
        
        train_loss = train(model, device, train_loader, criterion, optimizer, scaler, epoch)
        val_loss = validate(model, device, valid_loader, criterion)

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
        if epochs_without_improvement >= PATIENCE:
            print('\033[93m' + "Stopping early!" + '\033[0m')
            break

    # End TensorBoard Summary Writer
    writer.close()

    # Save the model
    torch.save(model, MODEL_SAVE_PATH)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt: # Terminate program when pressed CTRL+C
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)