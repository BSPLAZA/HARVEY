import torch
import torchvision.models as models
import os


def get_num_classes(root_dir):
    return len([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])


def modify_resnet50_for_dataset(model_path, save_path, dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model architecture without loading the weights
    model = models.resnet50(weights=None)

    # Modify the architecture BEFORE loading the state dictionary
    
    # The below line is for running the model on a GPU device.
    # num_classes_saved_model = torch.load(model_path)['fc.bias'].shape[0]
    # The below line is for running the model on a CPU device
    num_classes_saved_model = torch.load(model_path, map_location=device)['fc.bias'].shape[0]

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes_saved_model)

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Adjust the final layer to the desired number of classes
    num_features = model.fc.in_features
    num_classes = get_num_classes(dataset_path)
    model.fc = torch.nn.Linear(num_features, num_classes)

    # Save the modified model's state_dict
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    pretrained_model_path = '../../models/ResNet50-Plant-model-80.pth'
    modified_model_save_path = '../../models/ResNet50-Plant-model-80.pth'
    training_data_path = '../../.kaggle/train'

    modify_resnet50_for_dataset(
        pretrained_model_path, modified_model_save_path, training_data_path)
    print(f"Modified model saved to {modified_model_save_path}")
