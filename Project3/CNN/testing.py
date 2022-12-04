import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


# Load the CheXpert dataset
(X_train, y_train), (X_test, y_test) = tfds.load("chexpert", split=["train", "test"])


# Preprocess the data
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Flatten the data into a one-dimensional array
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Choose 5 random images from the dataset
num_images = 5
random_indices = np.random.choice(X_train.shape[0], size=num_images, replace=False)
random_images = X_train[random_indices]
random_labels = y_train[random_indices]

# Print the images and labels
for i in range(num_images):
    image = random_images[i]
    label = random_labels[i]
    
    # Reshape the image into its original 2D shape
    image = np.reshape(image, (150, 150))
    
    # Plot the image
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Print the label
    print(f"Label: {label}")
