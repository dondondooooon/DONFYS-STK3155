# Import necessary libraries
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# Download and preprocess the ChestX-ray14 dataset
# (This may involve resizing and normalizing the images,
# as well as splitting the dataset into training and validation sets)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.chestxray14.load_data()

# Define the architecture of the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(14, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the preprocessed ChestX-ray14 dataset
model.fit(X_train, y_train, epochs=10)

# Evaluate the model on the test set
model.evaluate(X_test, y_test)

# Use the trained model to make predictions on new images
predictions = model.predict(new_images)

# Randomly select 5 images from the ChestX-ray14 dataset
indices = random.sample(range(len(X_train)), 5)
images = X_train[indices]
labels = y_train[indices]

# Display the images and their labels
for image, label in zip(images, labels):
    plt.imshow(image)
    plt.show()
    print("Label: ", label)
