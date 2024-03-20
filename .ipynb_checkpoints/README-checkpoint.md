# MNIST Digit Recognition

## Overview
This project explores the use of deep learning models, specifically Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN), for the task of recognizing handwritten digits from the MNIST dataset. Through a series of exercises, the project demonstrates the impact of model architecture, hyperparameters, and evaluation metrics on the performance of neural networks.

## Features
- Implementation of a basic MLP model for digit recognition.
- Exploration of the effects of varying the number of neurons in the hidden layer.
- Analysis of the impact of batch sizes on model performance.
- Comparison of performance metrics: accuracy vs. mean square error (MSE).
- Advanced CNN model implementation with additional layers and features for improved accuracy.
- Systematic examination of convolutional kernel sizes and their effects on accuracy.
- Detailed study on the influence of the number of neurons in dense layers and the number of training epochs on model convergence and performance.
- Introduction of a more complex CNN architecture for enhanced digit recognition capabilities.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib (optional, for visualization)

## Dataset
The MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits, is used in this project. Each grayscale image is 28x28 pixels.

## Usage
1. Clone the repository to your local machine.
2. Ensure all required libraries are installed.
3. Run the `HandwrittenDigitRecognition.py` script to train and evaluate the models.
   - The script includes functions for training and predicting with both MLP and CNN models.
   - By default, the script executes the MLP model experiment. Modify the script to run CNN experiments as needed.

## Results
The project findings highlight the superiority of CNN models over MLP for image-based tasks, with detailed experiments showing how different aspects of the model architecture and training process influence the final model performance. Adjusting convolutional kernel sizes, the number of neurons in dense layers, and the number of training epochs were found to significantly affect classification error rates, demonstrating the need for careful model tuning.

## Conclusion
This project provides a comprehensive exploration of using neural networks for digit recognition, offering valuable insights into the design and optimization of deep learning models for image classification tasks. The experiments conducted underscore the importance of model architecture choices and hyperparameter tuning in achieving high model performance.

## Contributors
- Nizar ZEROUALE

