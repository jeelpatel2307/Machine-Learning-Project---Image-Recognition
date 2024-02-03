# Image Recognition with TensorFlow and Keras

This project demonstrates image recognition using a convolutional neural network (CNN) built with TensorFlow and Keras in Python. The model is trained on the CIFAR-10 dataset, and its performance is evaluated on a test set.

## Project Structure

### 1. `image_recognition.py`

- Contains the main code for image recognition.
- Uses TensorFlow and Keras to define, compile, and train a CNN.
- Loads and preprocesses the CIFAR-10 dataset for training and testing.
- Displays the training and validation accuracy over epochs and evaluates the model on the test set.

## How to Run

1. Save the code in a file named `image_recognition.py`.
2. Install required packages: `pip install tensorflow matplotlib`.
3. Run the program using the command:
    
    ```bash
    python image_recognition.py
    
    ```
    

## Usage

1. The program loads the CIFAR-10 dataset and preprocesses the images.
2. The CNN model is defined and compiled for training.
3. The model is trained on the training dataset for 10 epochs.
4. The test accuracy of the model is evaluated and printed.
5. Training and validation accuracy over epochs are visualized.

## Customization

Feel free to customize and expand upon this project based on your requirements. Possible enhancements include using different datasets, modifying the model architecture, experimenting with hyperparameters, and exploring transfer learning.

---
