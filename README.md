# **Handwritten Digit Recognition with CNN and MNIST**

This repository implements a **Convolutional Neural Network (CNN)** to predict handwritten digits using the famous **MNIST dataset**. 

MNIST is a standard dataset in the field of machine learning, containing **60,000 training images** and **10,000 test images** of handwritten digits (0 to 9). Each image is grayscale and has a resolution of **28x28 pixels**.

The primary goal of this project is to train a CNN model capable of accurately recognizing handwritten digits while exploring the fundamental concepts of CNNs in computer vision.

---

## **Key Features**

- **Dataset**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- **Model Architecture**: Convolutional Neural Network (CNN)
- **Programming Language**: Python
- **Main Libraries**:
  - `TensorFlow` / `Keras` for building and training the model.
  - `NumPy` for data manipulation.
  - `Matplotlib` for visualizing images and model performance.

---

## **Project Highlights**

1. **Data Loading and Preprocessing**:
   - Normalizing images (pixel values between 0 and 1).
   - Reshaping data to fit the format required by the CNN.

2. **CNN Model**:
   - Automatic extraction of local features from images using convolutional layers.
   - Feature map size reduction with pooling layers to improve robustness.
   - Classification of digits (0 to 9) using fully connected layers.

3. **Model Training and Validation**:
   - Training the model on the MNIST training dataset.
   - Evaluating performance on the test dataset.

4. **Prediction**:
   - Making predictions on new images.
   - Visualizing predictions and comparing them with true labels.

---

## **Model Architecture**

The CNN architecture used in this project includes the following components:

1. **Convolutional Layers (Conv2D)**:
   - First layer: 32 filters of size `3x3`, activation function `ReLU`.
   - Second layer: 64 filters of size `3x3`, activation function `ReLU`.

2. **Pooling Layers (MaxPooling2D)**:
   - Reducing the size of feature maps using a `2x2` pooling window.

3. **Flattening**:
   - Converting 2D feature maps into 1D vectors for fully connected layers.

4. **Fully Connected Layers (Dense)**:
   - One dense layer with 128 neurons, activation function `ReLU`.
   - Output layer with 10 neurons (one for each digit class), activation function `softmax`.

---

## **Expected Results**

- **Training Accuracy**: Approximately **99%** on MNIST.
- **Test Accuracy**: Approximately **98-99%**, demonstrating excellent generalization.

---

## **How to Run the Project?**

### **1. Prerequisites**
Ensure the following libraries are installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

You can install the required dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib
