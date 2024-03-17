# Handwritten Digit Classification with ANNs and CNNs

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Theoretical Background](#theoretical-background)
  - [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
- [Practical Implementation](#practical-implementation)
  - [Pre-requirements](#pre-requirements)
  - [Application Overview](#application-overview)
    - [ANN for Handwritten Digit Classification](#ann-for-handwritten-digit-classification)
    - [CNN for Handwritten Digit Classification](#cnn-for-handwritten-digit-classification)
- [Setup and Running Instructions](#setup-and-running-instructions)
- [Exercises](#exercises)
- [Contributing](#contributing)

## Overview
This project focuses on the classification of handwritten digits using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). Utilizing Python and TensorFlow with Keras, it implements a digit recognition framework to evaluate system performance using accuracy and loss metrics.

## Objectives
- To apply ANN and CNN architectures for image classification tasks.
- To implement a digit recognition model using TensorFlow and Keras.
- To evaluate and compare model performance using accuracy and loss.

## Theoretical Background
Understanding the distinction between ANNs and CNNs is crucial for modern image recognition tasks. This section elaborates on the scalability challenges faced by ANNs with large images and how CNNs overcome these through their specialized architecture.

### Convolutional Neural Networks (CNN)
CNNs leverage spatial hierarchies of features through convolutional layers, making them highly efficient for image processing. They utilize weight sharing and pooling to reduce the number of parameters, enabling a more scalable approach to image classification.

## Practical Implementation
### Pre-requirements
- Python 3.x, TensorFlow, and Keras should be installed.
- The `HandwrittenDigitRecognition.py` script is required in the project directory.

### Application Overview
#### ANN for Handwritten Digit Classification
- Load and preprocess the MNIST dataset.
- Implement and evaluate a simple MLP baseline model.

#### CNN for Handwritten Digit Classification
- Implement a CNN model suitable for digit classification.
- Preprocess data for CNN processing.
- Construct, train, and evaluate the CNN model.

## Setup and Running Instructions
1. Install the necessary Python packages:
   ```bash
   pip install tensorflow keras numpy
2. Execute the scripts for ANN and CNN models:
   python HandwrittenDigitRecognition_ANN.py
   python HandwrittenDigitRecognition_CNN.py

## Contributing
Contributions to improve or expand the project are welcome. Please adhere to the standard pull request process for submissions.


