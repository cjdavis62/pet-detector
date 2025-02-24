import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import ultralytics
from ultralytics import YOLO
from IPython.display import display, clear_output
import ipywidgets as widgets
import io
import numpy as np

def instantiate_model() -> ultralytics.models.yolo.model.YOLO:
    """Instantiates the model."""
    model = YOLO("yolov8m.pt")  # "n" is the nano version; use "m" or "l" for better accuracy
    return model


def print_detected(results: ultralytics.engine.results.Results) -> None:
    """Prints information about the observed objects"""

    # Extract detected objects
    detected_objects = results.boxes.cls.tolist()  # List of detected class indices
    class_names = results.names  # YOLO class names
    
    # Count unique objects
    from collections import Counter
    
    counts = Counter([class_names[int(idx)] for idx in detected_objects])
    for pet, num in counts.items():
        if pet not in ['cat', 'dog']:
            print(f"This {pet} is not a cat. Boring...")
        elif num > 1:
            print(f"{num} very cute {pet}s detected!")
        else:
            print(f"{num} very cute {pet} detected!")
        
        if pet == "cat":
            print("They're purrrrfect!")
        if pet == "dog":
            print("They're stinky, but still cute!")

def detect_objects(change):
    """Handles the image upload and runs YOLO inference"""
    clear_output(wait=True)
    # Get the uploaded image
    file_info = uploader.value[0]

    if not file_info:
        print("Please upload an image.")
        return
    # Read uploaded file

    file_name = file_info['name']
    file_bytes = file_info["content"]

    # Convert bytes to a NumPy image
    image = Image.open(io.BytesIO(file_bytes))
    image_cv = np.array(image)  # Convert to NumPy array

    # Convert RGB to BGR for OpenCV (YOLO expects BGR)
    #image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    model = instantiate_model()
    
    # Run YOLO inference
    results = model(image_cv)

    
    # Extract annotated image
    annotated_image = results[0].plot()


    # Convert BGR back to RGB for displaying
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


    
    cv2.imwrite("output.jpg", annotated_image)

    with output:
        display(Image.open("output.jpg"))

        # Print output about the detected objects
        print_detected(results[0]) 
    return


# Load YOLO model
model = instantiate_model()  # Instantiate your model here

# Create file uploader widget
uploader = widgets.FileUpload(accept='image/*', multiple=False)

# Output widget for displaying results
output = widgets.Output()

# Link the upload button to the function
uploader.observe(detect_objects, names='value')