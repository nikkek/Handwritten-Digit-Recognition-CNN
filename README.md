# Handwritten Digit Recognition with Convolutional Neural Network (CNN)

This repository contains scripts and resources for recognizing handwritten digits using Convolutional Neural Network (CNN). The project consists of three main components:

## 1. Data Preparation
The script in the `convert images to grayscale.py` converts the images from rgb to grayscale.

The `handwritten digits data preparation.py` arranges images into respective train and test directories.

## 2. Model Training
The `handwritten digits training.py` compiles the model, prepares image data generators, trains the model on training data, and evaluates its accuracy on test data.

## 3. Recognition
The `handwritten digits recognition.py` script uses a pre-trained CNN model to recognize digits in unidentified images. It reads images from the 'unidentified_images/' directory, predicts the digits, saves the recognized images in 'identified_images/' directory, and calculates recognition accuracy.

### Usage
- Ensure dependencies such as TensorFlow, Matplotlib, NumPy, and scikit-image are installed.

### Directory Structure
- `identified_images/`: Directory for saving recognized images.
- `images/`: Stores organized data for training inside `train/` and testing inside `test/` after preparation.
- `models/`: Stores saved models generated during training.
- `original_data`: Stores original images.
- `original_data_grayscale/`: Stores grayscale versions of original images.
- `plots/`: Stores plots of training history generated during model training.
- `unidentified_images/`: Location for unidentified images to be recognized.
