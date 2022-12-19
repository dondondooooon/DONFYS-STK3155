# Tuning Convolutional Neural Network (CNN) for CIFAR10 dataset
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

# Define etas for tuning
hp_etas = np.logspace(-4, 0, 5)

# Define lambdas for tuning
hp_lmbds = np.logspace(-4, 0, 5)

#Using Keras-tuner library to find the best hyperparameters for the CNN model
def build_model(hp):
    model = Sequential()

    lmb = hp.Choice('regularizer', values=hp_etas.tolist())
    learning_rate = hp.Choice('learning_rate', values=hp_lmbds.tolist())

    #model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Input(shape=(32, 32, 3)))

    for i in range(hp.Int('num_blocks', 1, 2)):
        hp_filters=hp.Choice('filters_'+ str(i+1), values=[32, 64])

        model.add(Conv2D(hp_filters, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(lmb)))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    hp_neurons = hp.Int('neurons', min_value=16, max_value=128, step=16)
    model.add(Dense(hp_neurons, activation='relu', kernel_regularizer=regularizers.l2(lmb)))

    model.add(Dense(10,activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'],
    )
    
    return model

# build_model(keras_tuner.HyperParameters())

tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=50,
    executions_per_trial=2,
    directory=".",
    project_name="tuning_cnn",
)

tuner.search(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
best_model = tuner.get_best_models()[0]

best_model.summary()

tuner.results_summary()

#Optimal Hyper-parameters as of 18/12/2022 - tuning-cnn-1
# {'regularizer': 0.0001, 'learning_rate': 0.0001,
#  'num_blocks': 1, <- need to 2x check this one -> seems like 2 layers give better results 
#  'filters_1': 32, 'filters_2': 64,
#  'neurons': 112}