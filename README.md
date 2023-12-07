# YOLOv8 Plant Disease Detection

## Overview

This repository contains code to train and execute a YOLOv8 model for plant disease detection using the PlantDoc and FieldPlant datasets. The implementation is provided in both Jupyter Notebook (`main.ipynb`) and Python script (`main.py`). Additionally, there is a `webcam.py` script that utilizes the trained model for live video prediction.

## Datasets

### PlantDoc Dataset
- [PlantDoc Dataset](https://universe.roboflow.com/joseph-nelson/plantdoc)
- Ensure to download the dataset in YOLOv8 format without overwriting the existing `data.yaml` file.

### FieldPlant Dataset
- [FieldPlant Dataset](https://universe.roboflow.com/plant-disease-detection/fieldplant)
- Similar to PlantDoc, download the dataset in YOLOv8 format without overwriting the `data.yaml` file.

## Files

### `main.ipynb`
- The Jupyter Notebook contains the latest code for training and executing the YOLOv8 model. Follow the step-by-step instructions and code cells for seamless implementation.

### `main.py`
- The Python script version of the notebook. Ensure to update this file with the latest code from `main.ipynb` for consistent execution.

### `webcam.py`
- This script allows for real-time prediction using the trained YOLOv8 model on live video. Simply run the script to initiate the webcam prediction.

## Usage

1. **Dataset Preparation**
   - Download the PlantDoc and FieldPlant datasets in YOLOv8 format.
   - Do not overwrite the existing `data.yaml` file.

2. **Training the Model**
   - Open and run the `main.ipynb` notebook or execute the updated code in `main.py`.
   - Follow the instructions to train the YOLOv8 model using the downloaded datasets.

3. **Live Video Prediction**
   - Run `webcam.py` to initiate live video prediction using the trained model.
   - Ensure the necessary dependencies are installed for webcam access.

## Dependencies
- The code is dependent on YOLOv8 and its associated libraries. Make sure to install the required dependencies by referring to the notebook or script.

## Note
- Keep the datasets separate and do not overwrite the `data.yaml` file when downloading the datasets.

## Contributors
- Add your name if you've contributed to this project.

## License
- Mention the license under which this project is distributed.

Feel free to reach out for any issues or improvements! Happy coding!
