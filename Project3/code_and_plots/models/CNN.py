# Convolutional Neural Network (CNN) for CIFAR10 dataset
# Import necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import datasets
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import Sequential      #This allows appending layers to existing models
from keras.layers import Flatten, Dense, Dropout, Conv2D, Input, BatchNormalization, MaxPooling2D  #This allows defining the characteristics of a particular layer
from keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import keras_tuner as kt


# Initialize random seed with np 
np.random.seed(2022)
sns.set()

# Set figure dimensions to avoid scaling in LaTeX.
def set_size(width, fraction=1):
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # Manual addons
    heightadd = inches_per_pt * 45
    widthadd = inches_per_pt * 65
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt + widthadd
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio + heightadd
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim
    
# Load the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Adding validation set
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size = 0.2, random_state=1) 

# Normalize the pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0
val_images = val_images / 255.0

# one-hot encode the labels
# train_labels = keras.utils.to_categorical(train_labels, 10)
# test_labels = keras.utils.to_categorical(test_labels, 10)
# val_labels = keras.utils.to_categorical(val_labels, 10)

# Define the class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# # Select 16 random images from the training set
# num_images = 16
# def plot_images(num_images):
#     plt.figure(figsize=(10,10))
#     for i in range(num_images):
#         plt.subplot(4,4,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(train_images[i], cmap=plt.cm.binary)
#         # The CIFAR labels happen to be arrays, 
#         # which is why you need the extra index
#         plt.xlabel(class_names[train_labels[i][0]])
#     plt.show()

# plot_images(num_images)


# # Print the selected images and their labels
# for i in range(5):
#     print("Image #{}: label = {}".format(i+1, selected_labels[i]))
#     plt.imshow(selected_images[i])
#     plt.show()

# Chosen lambda
lmb = 0.0001
# Chosen learning rate
learning_rate = 0.0001

# # Define the model
model = Sequential()
# Input layer
model.add(Input(shape=(32, 32, 3)))

# Block 1  
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(lmb)))
# batch normalization
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# # Block 2
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(lmb)))
# batch normalization
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
# batch normalization
model.add(Dropout(0.25))

# Flatten the output and feed it into a hidden dense layer
model.add(Flatten())
model.add(Dense(112, activation='relu', kernel_regularizer=regularizers.l2(lmb)))

# Output layer
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))

# Compile the model
model.compile(tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=100, 
                    validation_data=(val_images, val_labels), batch_size=32)

# Extracting the training and validation accuracy values for each epoch
acc_log = history.history['accuracy']

# Extract the epoch numbers
epoch_num = range(1, len(acc_log) + 1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# Plot the accuracy vs epochs graph
# Summarize history for accuracy
plt.style.use("ggplot")
plt.figure(figsize=set_size(345), dpi=80)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.savefig(f"acc_epoch.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

# Summarize history for loss
plt.style.use("ggplot")
plt.figure(figsize=set_size(345), dpi=80)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.savefig(f"loss_epoch.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

# Print the error metrics
print("\n--------------Error metrics for the Convolutional Neural Network (CNN) model:--------------")
# Print the final training and validation accuracy
print("\nFinal training accuracy:", history.history['accuracy'][-1])
# Print the final training and validation loss
print("Final training loss:", history.history['loss'][-1])

# Print the test accuracy
print("\nTest accuracy:", test_acc)
# Print the test loss
print("Test loss:", test_loss)
print("---------------------------------------------------------------------------------------------")