import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
model = load_model("dog_cat_classifier_vgg16.h5")

# Image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Function to show the image
def show_image(image_path):
    # Open and display the image in high quality
    img = Image.open(image_path)
    img.show()

# Function to predict the accuracy of the image
def predict_accuracy(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)
    
    # Print prediction results with accuracy
    if prediction[0][0] > 0.5:
        accuracy = prediction[0][0] * 100
        print(f"The image at {image_path} is a Dog ({accuracy:.2f}% confidence)")
    else:
        accuracy = (1 - prediction[0][0]) * 100
        print(f"The image at {image_path} is a Cat ({accuracy:.2f}% confidence)")

# Example usage
unknown_image_path = r"C:\Users\soumy\Desktop\test_img\cat2.jpg"

# First, show the image
show_image(unknown_image_path)

# Then, predict the accuracy
predict_accuracy(unknown_image_path)
