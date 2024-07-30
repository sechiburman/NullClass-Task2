import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications import MobileNetV2   # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # type: ignore

# Load YOLO model and configuration files
net = cv2.dnn.readNet("D:/vscode/yolov3.weights", "D:/vscode/yolov3 (1).cfg")
with open("D:/vscode/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load the car color classification model
car_color_model = load_model('car_color_model_best.keras')
class_labels = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

class GenderClassifier:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, image):
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        gender = 'male' if prediction > 0.5 else 'female'
        return gender

    def preprocess_image(self, image):
        resized_image = tf.image.resize(image, (224, 224))
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0)

def extract_person_image(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]

def preprocess_car_image(car_img, target_size=(64, 64)):
    image = cv2.resize(car_img, target_size)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

def predict_car_color(car_color_model, car_img, class_labels):
    image = preprocess_car_image(car_img)
    prediction = car_color_model.predict(image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Streamlit app
st.title("Traffic Signal Image Analysis")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load input image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")
    
    # Create a 4D blob from the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    
    # Set input blob for the network
    net.setInput(blob)
    
    # Forward pass through the network to get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    
    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Get bounding box coordinates
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression to eliminate redundant overlapping boxes with lower confidences
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Initialize counters
    car_count = 0
    male_count = 0
    female_count = 0
    other_vehicle_count = 0
    
    # Create an instance of GenderClassifier
    gender_classifier = GenderClassifier()
    
    # Draw bounding boxes and labels on the image
    if isinstance(indices, np.ndarray) and indices.ndim == 2:
        indices = indices.flatten()

    car_colors = []

    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        label = str(classes[class_ids[i]])
    
        if label == "car":
            car_count += 1
            car_img = image[y:y + h, x:x + w]
            color_name = predict_car_color(car_color_model, car_img, class_labels)
            car_colors.append(color_name)
    
            # Swap red and blue colors for visualization
            if color_name.lower() == "red":
                color_bgr = (255, 0, 0)  # Red to blue
                swapped_color_name = "blue"
            elif color_name.lower() == "blue":
                color_bgr = (0, 0, 255)  # Blue to red
                swapped_color_name = "red"
            else:
                color_bgr = (0, 255, 0)  # Green for other colors
                swapped_color_name = color_name

            st.write(f"Detected car color: {color_name}")
            cv2.rectangle(image, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(image, f"{label}, {swapped_color_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        elif label in ["bus", "truck", "motorbike", "bicycle"]:
            other_vehicle_count += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif label == "person":
            person_img = extract_person_image(image, (x, y, x + w, y + h))
            if person_img.size > 0:
                gender = gender_classifier.predict(person_img)
                if gender == "male":
                    male_count += 1
                else:
                    female_count += 1
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(image, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Display results
    st.image(image, channels="BGR", caption="Processed Image")
    st.write(f"Car count: {car_count}")
    st.write(f"Male count: {male_count}")
    st.write(f"Female count: {female_count}")
    st.write(f"Other vehicle count: {other_vehicle_count}")
    st.write(f"Car colors: {', '.join(car_colors)}")
