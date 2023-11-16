import cv2
import numpy as np
from time import time
from ultralytics import YOLO

class WebcamObjectDetection:
    def __init__(self, model_path, class_names):
        # Initialize the webcam. '0' denotes the default webcam.
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")  # Error handling if webcam cannot be accessed.

        # Load the YOLO model from the specified path.
        self.model = YOLO(model_path)

        # Store the class names for object detection.
        self.class_names = class_names

    def predict(self, frame):
        # Use the YOLO model to predict objects in the frame.
        return self.model(frame)

    def plot_bboxes(self, results, frame):
        # Extract bounding boxes, confidences, and class IDs from the results.
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        # Draw bounding boxes and labels on the frame.
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = confidences[i]
            cls_id = class_ids[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Draw rectangle.
            label = f'{self.class_names[cls_id]} {conf:.2f}'  # Create label with class name and confidence.
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Put text on frame.

        return frame

    def __call__(self):
        # Continuously capture frames from the webcam and process them.
        while True:
            start_time = time()
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Predict and plot bounding boxes on the frame.
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            # Calculate and display the frames per second (FPS).
            end_time = time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Display the processed frame.
            cv2.imshow('Webcam', frame)
            # Break the loop if 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close all OpenCV windows.
        self.cap.release()
        cv2.destroyAllWindows()

# Usage example.
model_path = '../../weights/best.pt'  # Ensure this path to the model weights is correct.
class_names = ['Cassava Bacterial Blight', 'Cassava Brown Leaf Spot', 'Cassava Healthy', 'Cassava Mosaic', 'Cassava Root Rot', 'Corn Brown Spots', 'Corn Charcoal', 'Corn Chlorotic Leaf Spot', 'Corn Gray leaf spot', 'Corn Healthy', 'Corn Insects Damages', 'Corn Mildew', 'Corn Purple Discoloration', 'Corn Smut', 'Corn Streak', 'Corn Stripe', 'Corn Violet Decoloration', 'Corn Yellow Spots', 'Corn Yellowing', 'Corn leaf blight', 'Corn rust leaf', 'Tomato Brown Spots', 'Tomato bacterial wilt', 'Tomato blight leaf', 'Tomato healthy', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus']
detector = WebcamObjectDetection(model_path, class_names)
detector()  # Start the object detection.