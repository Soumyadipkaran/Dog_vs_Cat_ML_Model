
# Dog vs. Cat Classifier Using VGG16

This project implements a binary image classifier to classify images of dogs and cats using a deep learning model based on **VGG16** architecture with transfer learning. The model is trained using TensorFlow/Keras and is capable of predicting whether an image contains a dog or a cat.

## Necessary Libraries and Installation

This project requires the following Python libraries. You can install them using `pip`:

- **TensorFlow**: Used for building and training deep learning models.
  ```bash
  pip install tensorflow
  ```

- **NumPy**: Provides support for large multidimensional arrays and matrices.
  ```bash
  pip install numpy
  ```

- **Matplotlib**: A plotting library for visualizing training progress and metrics.
  ```bash
  pip install matplotlib
  ```

- **Keras** (part of TensorFlow): High-level API used for building and training neural networks.
  - Keras is bundled with TensorFlow, so installing TensorFlow also installs Keras.

- **OS**: Used for interacting with the file system and paths (this is a built-in Python module and doesn't need installation).

- **ImageDataGenerator**: Part of Keras, used to generate batches of image data with real-time data augmentation.

## Directory Setup

You need to organize your data into directories like so:

```
- Dog_Cat/
    - train/
        - dogs/
        - cats/
    - test/
        - dogs/
        - cats/
```

- **`train/`**: Contains subdirectories for each class (e.g., `dogs/` and `cats/`), with images of each class for training.
- **`test/`**: Contains subdirectories for each class (e.g., `dogs/` and `cats/`), with images of each class for testing.

Update the following paths in the code to match where your images are stored:

```python
train_dir = "C:\path\to\your\train"
test_dir = "C:\path\to\your\test"
```

Ensure the image dimensions in the code (`IMG_HEIGHT = 150, IMG_WIDTH = 150`) are appropriate for your dataset.

## Model Logic and Workflow

This code trains a deep learning model to classify images of dogs and cats using a **VGG16** architecture as the base model, with additional layers added on top.

### Step-by-Step Explanation:

1. **Importing Libraries**:
   - The code begins by importing the necessary libraries: TensorFlow/Keras for building and training the model, Matplotlib for plotting the training history, and OS/NumPy for handling file paths and arrays.

2. **Defining Image Dimensions and Batch Size**:
   - The image dimensions (`IMG_HEIGHT = 150, IMG_WIDTH = 150`) are set to resize input images.
   - `BATCH_SIZE = 32` defines the number of samples processed before the model is updated.

3. **Data Augmentation & Preprocessing**:
   - `ImageDataGenerator` is used for real-time data augmentation during training. The training images are rescaled by dividing pixel values by 255, and several augmentation techniques (such as rotation, flipping, zooming, etc.) are applied to improve generalization.
   - The test images are only rescaled (without augmentation) to evaluate the model.

4. **Loading Data**:
   - `train_generator` and `test_generator` are created using `ImageDataGenerator.flow_from_directory()` to load images from the respective directories (`train_dir` and `test_dir`).
   - The `class_mode='binary'` option specifies a binary classification task (dog vs. cat), and the images are resized to the target size of `(150, 150)`.

5. **VGG16 Model Setup**:
   - A pre-trained **VGG16** model (trained on ImageNet) is loaded as the base model using `VGG16(weights='imagenet', include_top=False)`. 
     - `include_top=False`: Excludes the top (fully connected) layers of VGG16 since we will add our custom layers for classification.
     - `input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)` specifies that the input images will have 3 color channels (RGB) and the size `(150, 150)`.

6. **Freezing Pre-trained Layers**:
   - The layers of the pre-trained VGG16 model are frozen (set to `trainable=False`), so their weights are not updated during training. This prevents overfitting and reduces computational cost, allowing the model to leverage the features learned from ImageNet.

7. **Building the Model**:
   - A **Sequential model** is defined with the following layers:
     - `base_model`: VGG16 as the feature extractor.
     - `Flatten()`: Converts the 2D features into 1D.
     - `Dense(512, activation='relu')`: A fully connected layer with 512 units and ReLU activation for non-linearity.
     - `Dropout(0.5)`: Regularization technique to prevent overfitting by randomly setting 50% of the inputs to 0 during training.
     - `Dense(1, activation='sigmoid')`: The output layer with a single unit for binary classification (dog or cat), using the sigmoid activation function to output a value between 0 and 1 (representing probability).

8. **Compiling the Model**:
   - The model is compiled with the Adam optimizer and binary cross-entropy loss function (`binary_crossentropy`), appropriate for binary classification. The `accuracy` metric is used to track model performance.

9. **Early Stopping**:
   - An **EarlyStopping** callback is used to stop training if the validation loss does not improve for 5 consecutive epochs (`patience=5`). This prevents overfitting and ensures the best model is retained.

10. **Training the Model**:
    - The model is trained using `model.fit()`, which uses the `train_generator` for training and `test_generator` for validation. The training history is saved in the `history` object.

## Saving the Model

After training, the model is saved to a file using:

```python
model.save("dog_cat_classifier_vgg16.h5")
```

This saves the entire model, including the architecture, weights, and optimizer state, to a `.h5` file. You can later load this model using `tf.keras.models.load_model()` for inference or further training.

## Evaluating the Model

After training, the model is evaluated on the test data:

```python
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

This evaluates the model on the test set and prints the test accuracy as a percentage.

## Plotting the Training History

Finally, a function `plot_training_history()` is defined and used to plot the training and validation accuracy and loss over the epochs:

```python
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
```

This will generate two plots:
- One showing the training and validation accuracy over epochs.
- Another showing the training and validation loss over epochs.

## Conclusion

This project implements a binary classification model to classify images of dogs and cats using transfer learning with the **VGG16** model. The model is trained with data augmentation techniques to improve generalization, and the performance is evaluated on a test dataset. The trained model is saved as `dog_cat_classifier_vgg16.h5`, and visualizations of the training process are provided.

You can further fine-tune the model, change the architecture, or use it for inference by loading the saved model.


# Test the Dog vs. Cat Classifier

This script allows you to test the trained **Dog vs. Cat Classifier** with unknown images. You can load an image, display it, and predict whether the image contains a dog or a cat, along with the confidence score.

## **Necessary Libraries and Installation**

Before running the test, make sure to install the following libraries:

- **TensorFlow**: Used to load and use the trained deep learning model.
  ```bash
  pip install tensorflow
  ```

- **NumPy**: A library for handling arrays and matrices.
  ```bash
  pip install numpy
  ```

- **Pillow**: Used for opening and displaying images.
  ```bash
  pip install pillow
  ```

---

## **Testing Process**

### **1. Import the Model**

Load the pre-trained model from the file `dog_cat_classifier_vgg16.h5`. This file contains the model's architecture and weights, which were previously trained.

### **2. Image Path**

Specify the path of the image you want to test. Update the file path in the script for the image you want to test.

- Example:
  ```plaintext
  unknown_image_path = r"C:\Users\soumy\Desktop\test_img\cat1.jpeg"
  ```

### **3. Display the Image**

The script will open and display the image so you can visually verify which image is being tested.

### **4. Prediction**

The model will predict whether the image contains a **dog** or a **cat** and display the confidence score in percentage.

- For example:
  ```plaintext
  The image at C:\Users\soumy\Desktop\test_img\cat1.jpeg is a Dog (87.45% confidence)
  ```

---

## **Test Case Images**

Below are two examples of test cases:

- **Test Case 1: Cat Image**

  Path: `C:\Users\soumy\Desktop\test_img\cat1.jpeg`

- **Test Case 2: Dog Image**

  Path: `C:\Users\soumy\Desktop\test_img\dog1.jpeg`

---

This README provides the details for testing the model with any unknown image of dogs or cats, as long as the paths are correctly set.

