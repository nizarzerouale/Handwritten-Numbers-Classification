import keras
import tensorflow as tf
from keras.datasets import mnist
import np_utils
from keras import layers
from keras.src.utils.np_utils import to_categorical
import numpy as np

# Assuming the baseline_model function is defined elsewhere in your script
from HandwrittenDigitRecognition import baseline_model

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Preprocess the test data: Flatten, normalize
num_pixels = X_test.shape[1] * X_test.shape[1]
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') / 255
Y_test = to_categorical(Y_test)
num_classes = Y_test.shape[1]

# Rebuild the model architecture
model = baseline_model(num_pixels, num_classes)

# Load the saved weights
model.load_weights('mnist_model_weights.h5')

# Select the first 5 images from the test dataset
X_sample = X_test[:5]
Y_sample = Y_test[:5]

# Make predictions
predictions = model.predict(X_sample)

# Convert the predictions from probability vectors to class labels
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(Y_sample, axis=1)

# Display the predictions and the actual labels
for i, (predicted, actual) in enumerate(zip(predicted_classes, actual_classes)):
    print(f"Image {i}: Predicted = {predicted}, Actual = {actual}")