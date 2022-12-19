# Feed-Forward Neural Network (FFNN) for CIFAR10 dataset
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
from keras.layers import Flatten, Dense, Dropout  #This allows defining the characteristics of a particular layer
from keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import keras_tuner as kt

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
    
'''
Data set - CIFAR10
60,000 colored 32x32 pixels images in 10 classes, with 6,000 images in each class. 
The dataset is divided into 50,000 training images and 10,000 testing images. 
The classes are mutually exclusive and there is no overlap between them. (Non multi-label)
1. Import data set // split for train, validation, and test
2. One-Hot Representation of Labels / Targets
3. Normalize pixel value between 0 and 1 
4. Verifying data set & Checking Dimensions
'''
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size = 0.2, random_state=1)
train_labels, test_labels, val_labels = to_categorical(train_labels), to_categorical(test_labels), to_categorical(val_labels)
train_images, test_images, val_images = train_images / 255.0 , test_images / 255.0 , val_images / 255.0


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# print(train_images.shape) # 40000 x 32 x 32 x 3
# print(test_images.shape) # 10000 x 32 x 32 x3 
# print(val_images.shape) # 10000 x 32 x 32 x3 
# print(train_labels.shape) # 40000 x 10
# print(test_labels.shape) # 10000 x 10
# print(val_labels.shape) # 10000 x 10

'''
Model Building
1. Define hyperparameters eta & lambda for tuning
2. Prepare for architectural tuning of the network (max 2 hidden layers)
Network uses Categorical Cross-Entropy for Loss function,
Accuracy for Cost function, ADAM optimization and Dropout Layer of p = 0.5
'''
def create_ffnn(n_neurons,activations,eta,lmbd):
    model = Sequential()
    # Input Layer
    model.add(Flatten(input_shape = (32,32,3)))
    model.add(Dense(n_neurons[0], input_shape = (3072,), activation=activations[0], kernel_regularizer=regularizers.l2(lmbd)))

    # Hidden Layer + Dropout Layer
    model.add(Dense(n_neurons[1], activation = activations[1], kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(n_categories, activation = "softmax")) # n_categories = 10

    # Compile Model
    model.compile(optimizer = optimizers.Adam(learning_rate = eta), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

n_categories = train_labels.shape[1] 
best_eta = 0.01 # best result from tuning
best_lmbd = 0.0001 # best result from tuning
n_neurons, activations = [256,256,224], ['relu', 'sigmoid', 'relu'] # best result from tuning
n_neurons = np.array(n_neurons)
activations = np.array(activations)
epoch = 100
batchsize = 32

FFNN = create_ffnn(n_neurons,activations,best_eta,best_lmbd)
history = FFNN.fit(train_images, train_labels, epochs=100, 
                    validation_data=(val_images, val_labels), batch_size=32)
scores = FFNN.evaluate(test_images,test_labels, verbose = 2) # dimensions 10x1 or 10, 

print("\nTest Accuracy:", scores[1])
print("Test Loss:", scores[0])

# # list all data in history
# print(history.history.keys())

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