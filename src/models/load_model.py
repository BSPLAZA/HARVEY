import torch
import torchvision.models as models


def load_resnet50_model(path='../../models/ResNet50-Plant-model-80.pth', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model architecture
    model = models.resnet50(pretrained=False)

    # Load the state dictionary
    state_dict = torch.load(path, map_location=device)

    # Update the final layer to match the saved model's number of output units
    num_classes = state_dict['fc.weight'].shape[0]
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(state_dict)

    return model
