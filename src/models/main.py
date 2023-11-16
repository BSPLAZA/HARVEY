import requests
import cv2
import matplotlib.pyplot as plt
import glob 
import random
import os
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
import clearml


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
OUTPUT_MODEL_NAME = 'plant_detection_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)

        thickness = max(2, int(w/275))
                
        cv2.rectangle(
            image, 
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
    return image


# Function to plot images with the bounding boxes.
def plot(image_paths, label_paths, num_samples):
    all_images = []
    all_images.extend(glob.glob(image_paths+'/*.jpg'))
    all_images.extend(glob.glob(image_paths+'/*.JPG'))
    
    all_images.sort()

    num_images = len(all_images)
    
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image_name = all_images[j]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[j])
        with open(os.path.join(label_paths, image_name+'.txt'), 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.lstrip().split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')

    plt.subplots_adjust(wspace=1)
    plt.tight_layout()
    plt.show()


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def autosplit(path, weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', ]  # 2 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file


# Runs model test on four ramdom images from directory 
def showTest(image_folder, m):
    
    image_paths = glob.glob(f'{image_folder}/*.jpg')  # Adjust the pattern if necessary
    random_image_paths = random.sample(image_paths, min(len(image_paths), 4))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Adjust the figsize if necessary
    axes_flat = axes.flatten()

    # Run inference and show results for each of the four random images
    for idx, image_path in enumerate(random_image_paths):
        results = m(image_path)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # convert to RGB PIL image
            axes_flat[idx].imshow(im)
            axes_flat[idx].axis('off')  # Hide the axis

    plt.tight_layout()
    plt.show()


def main(): 
    
    # Setup connection for ClearML
    clearml.browser_login()

    # Visualize a few training images.
    plot(
        image_paths = './datasets/FieldPlant/train/images', 
        label_paths = './datasets/FieldPlant/train/labels',
        num_samples=4,
    )

    SplitPath = './datasets/FieldPlant/train/images'

    # Split dataset into train, val, test
    autosplit(SplitPath)

    # Load generic yolo model
    model = YOLO('./yolov8n.pt')

    # Train the model
    results = model.train(
        data = 'C:/Users/shaun/Documents/capstone/plant_disease_detection/datasets/FieldPlant/data.yaml',
        imgsz = 1280,
        epochs = 250,
        batch = 32, # Use -1 for autobatch if you are unsure of your vram size
        cache = 'ram', # comment this out if you don't have at least 64gb of ram
        name = OUTPUT_MODEL_NAME
    )

    # Load newly trained model
    model = YOLO('./runs/detect/' + OUTPUT_MODEL_NAME + '/weights/best.pt')

    metrics = model.val() # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

    # Run inference on validation set
    results = model.predict(
        source = './datasets/FieldPlant/train/autosplit_val.txt',
        imgsz = 1280,
        name = OUTPUT_MODEL_NAME + '_infer',
        show_labels = False
    )

    # Runs model test on four ramdom images from directory 
    showTest(SplitPath, model)


if __name__=="__main__": 
    main() 