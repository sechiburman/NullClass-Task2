# Traffic Signal Image Analysis

This project is a Streamlit application that analyzes images of traffic scenes to detect and classify cars, their colors, and the gender of individuals. It uses the YOLO object detection model and a custom car color classification model.

## Features

- Upload and analyze images of traffic scenes.
- Detect and classify cars, their colors, and the gender of individuals.
- Display the number of cars, other vehicles, males, and females in the image.
- Display the detected car colors, swapping red and blue for visualization purposes.

## Requirements

The required Python packages are listed in `requirements.txt`. These packages must be installed in the system for the successfull running of the source code file (final.ipynb and final.py).

## Additional files required: 

- YOLO weights file (yolov3.weights)
- YOLO configuration file (yolov3.cfg)
- COCO class labels file (coco.names)
- Pre-trained car color classification model ('car_color_model_best.keras') (included in this repository)
- VCoR dataset for training the car color model (link provided below)

## Running the Application

To run the application, use the following command:
streamlit run final.py

## Detailed Explanation of the Code
1. Loading Models and Configurations:
YOLO model and configuration files are loaded.
COCO class labels are read from a file.
A pre-trained car color classification model is loaded.

2. Gender Classifier:
The GenderClassifier class uses MobileNetV2 to predict the gender of individuals.

3. Image Preprocessing and Prediction:
The uploaded image is read and processed.
YOLO model detects objects in the image and applies non-maxima suppression.
Detected objects are classified as cars, other vehicles, or persons.
Car colors are predicted using the car color classification model.
The gender of individuals is predicted using the gender classifier.

4. Streamlit Application:
Users can upload an image using the file uploader.
The uploaded image is processed and results are displayed, including bounding boxes, labels, and counts of detected objects.

## Testing with Provided Images (Using final.ipynb)
1. Open and Configure the Notebook:
Open final.ipynb in Jupyter Notebook or Jupyter Lab.
Update the image paths in the notebook to point to the provided images for testing.

2. Run the Notebook:
Execute each cell in the notebook sequentially.
The notebook will load the images, perform object detection using YOLO, classify car colors using the pre-trained model, and predict genders using MobileNetV2.

3. View Results:
The notebook will display the processed images with bounding boxes, labels, and predictions for each image.

## Car Color Model
The car color classification model is trained using the code provided in the VCoR.ipynb notebook. The pre-trained model is saved as car_color_model_best.keras and is used in this application to classify car colors.

VCoR Dataset
The VCoR dataset is used to train the car color classification model. This dataset is not included in the repository and must be obtained separately. The dataset includes images of cars labeled with their respective colors.
Link for the dataset: https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset

1. Training the Car Color Model
To understand how the car color classification model was trained, refer to the VCoR.ipynb notebook in the repository. This notebook includes the data preprocessing steps, model architecture, training process, and evaluation metrics.

2. Using the Pre-trained Model
The pre-trained model file car_color_model_best.keras is included in the repository. This model is loaded in the Streamlit application to predict car colors.

## Usage
1. Uploading an Image:
Use the file uploader to select an image file (JPG, JPEG, PNG).
2. Displaying Results:
The application will display the uploaded image with bounding boxes and labels for detected objects.
It will also display the number of cars, other vehicles, males, females, and the detected car colors.

## Example
After running the application, you will see a Streamlit interface with an option to upload an image. Once an image is uploaded, the application will analyze it and display the results as described above.

## Note
This application requires pre-trained models and configuration files for proper functioning. Ensure that these files are correctly specified in the code.
