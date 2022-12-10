# Imports 
import os
import random
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# Set the path to the folder containing the images
image_path =  r"C:\Users\aliva\.keras\datasets\chestxray14"


# Define the percentage of images to use for the training set
train_percentage = 0.8

# Load the images
images = []
labels = []
for image_name in os.listdir(image_path):
  # Load the image and its associated label
  image = plt.imread(os.path.join(image_path, image_name))
  label = int(image_name.split("_")[0])

  # Add the image and its label to the list of images and labels
  images.append(image)
  labels.append(label)

# Filter out images that don't have the same shape as the first image
image_shape = images[0].shape
filtered_images = []
for image in images:
  if image.shape == image_shape:
    filtered_images.append(image)
labels = np.array(labels)

# Print the number of images left after filtering
print("\nNumber of images after filtering:", len(filtered_images), "\n")

# Shuffle the images and labels
indices =  []
indices = np.array(indices, dtype=int)
print("Indices before shuffle:", indices)
indices = np.random.permutation(len(images))
indices = indices.flatten()  # convert to one-dimensional array of integers
indices = np.clip(indices, 0, None) # remove negative indices
indices = np.minimum(indices, len(images) - 1) # remove indices that are too large
print("Indices after shuffle:", indices)
# Check the shape of the indices array
print("Indices shape:", indices.shape)
images = images[indices]
labels = labels[indices]

# Print the number of images left after filteringv2
print("\nNumber of images after filteringv2 twice:", len(filtered_images), "\n")


# Split the images into training and test sets
train_size = int(len(images) * train_percentage)
train_images = images[:train_size]
train_labels = labels[:train_size]
test_images = images[train_size:]
test_labels = labels[train_size:]

# Build the neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Compile the model
model.compile(
  optimizer="adam",
  loss="sparse_categorical_crossentropy",
  metrics=["accuracy"],
)

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Select 5 random images from the training data
indices = random.sample(range(len(train_images)), 5)

# Display the selected images and their labels
for index in indices:
  plt.imshow(train_images[index])
  plt.title(train_labels[index])
  plt.show()

   
