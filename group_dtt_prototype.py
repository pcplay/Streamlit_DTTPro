import streamlit as st
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
import random

# Streamlit title
st.title("Improved People Detection with YOLOv5")

# Load the YOLOv5 model
@st.cache_resource
def load_yolov5_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)  # Load the small version of YOLOv5
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_yolov5_model()

# Detect people in a given image using YOLOv5
def detect_people_yolo(image):
    """
    Detect people in a given image using YOLOv5.
    
    Args:
    - image: Input image in PIL format.
    
    Returns:
    - Image with bounding boxes drawn around detected people and a count of people.
    """
    results = model(image)  # Perform inference
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    np_img = np.array(image)  # Convert the image to a numpy array for drawing bounding boxes
    
    h, w, _ = np_img.shape
    people_count = 0  # Counter for detected people
    group_id = random.randint(1, 100)  # Generate a random group ID between 1 and 100

    for i, label in enumerate(labels):
        if int(label) == 0:  # '0' is the class for 'person' in YOLOv5
            x_min, y_min, x_max, y_max, confidence = cords[i]
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)
            cv2.rectangle(np_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(np_img, f"Person {confidence:.2f}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            people_count += 1  # Increment the people count

    # Display the count of people detected and their group information
    st.write(f"Number of people detected: {people_count}")
    st.write(f"Group ID assigned to this detection: {group_id}")

    # Ask for confirmation on the number of detected people
    confirm_count = st.number_input("Confirm the number of people detected:", min_value=0, max_value=100, value=people_count)
    
    if confirm_count != people_count:
        st.warning("The detected number of people differs from your confirmation.")

    return np_img

# Main function to run the Streamlit app
def main():
    """
    Main function to take a screenshot from the camera, detect people using YOLOv5, and display the results.
    """
    image_input = st.camera_input("Take a picture")  # Use Streamlit's camera input for taking pictures

    if image_input is not None:
        image = Image.open(image_input)  # Convert the image into a PIL format

        detected_image = detect_people_yolo(image)  # Detect people in the captured image using YOLOv5
        st.image(detected_image, caption="Detected People", use_column_width=True)  # Display the image

if __name__ == '__main__':
    main()