# Tuning Feed-Forward Neural Network (FFNN) for CIFAR10 dataset
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
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Flatten, Dense, Dropout  #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import keras_tuner as kt

np.random.seed(2022)
sns.set()
    
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


# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']
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


n_categories = train_labels.shape[1] # = 10
etas = np.logspace(-4,0,5)
lmbds = np.logspace(-4,0,5)
print("etas:", etas)
print("lambdas:", lmbds)

# Build FFNN Model
def build_model(hp):
    hp_learning_rate = hp.Choice('learning_rate', values = etas.tolist())
    hp_lmb_regularizer = hp.Choice('regularizer', values = lmbds.tolist())
    activation_choice_in = hp.Choice(name = 'activation_input', values=['sigmoid','relu'])
    adamopt = optimizers.Adam(lr = hp_learning_rate)

    model = Sequential()
    # Input Layer
    model.add(Flatten(input_shape = (32,32,3)))
    model.add(Dense(units = hp.Int('input_l', min_value = 32, max_value = 256, step = 32), input_shape = (3072,),\
        activation=activation_choice_in, kernel_regularizer=regularizers.l2(hp_lmb_regularizer)))

    # Hidden Layer + Dropout Layer
    for i in range(hp.Int('num_hidden_layers', 1, 2)):
        activation_choice_h = hp.Choice(name='activation_h'+ str(i), values=['sigmoid','relu'])
        model.add(Dense(units = hp.Int('hidden_' + str(i), min_value = 32, max_value = 256, step = 32),\
            activation = activation_choice_h, kernel_regularizer=regularizers.l2(hp_lmb_regularizer)))
        model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(n_categories, activation = "softmax")) # n_categories = 10

    # Compile Model
    model.compile(optimizer = adamopt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


# Tune Network
tuner_ffnn = kt.BayesianOptimization(
    build_model,
    objective = 'val_accuracy',
    max_trials = 50,
    executions_per_trial = 2,
    directory = '.',
    project_name = 'ffnn_tuning')

tuner_ffnn.search(train_images, train_labels, epochs=50, batch_size=32, validation_data=(val_images, val_labels))#, callbacks=callback)

best_ffnn_hyperparameters = tuner_ffnn.get_best_hyperparameters(1)[0]
print("Best Hyper-parameters")
best_ffnn_hyperparameters.values

print("\n\n")
print("Dis\n")
# Get top 2 models
models = tuner_ffnn.get_best_models(num_models=2)
best_model = models[0]
# Build model
# Needed for 'sequential' without specified 'input_shape'
best_model.build(input_shape=(None,32,32,3))
best_model.summary()

#Optimal Hyper-parameters as of 18/12/2022 - tuning-cnn-1
# {'regularizer': 0.0001, 'learning_rate': 0.01,
#  'activation_input': relu,
#  'input_l': 256,
#  'num_hidden_layers': 1,
#  'activation_h0': sigmoid, 
#  'hidden_0': 256,
#  'activation_h1': relu, <--- ignored cuz opt num_hidden_layers = 1 ; also tried to add it, gave worse results 
#  'hidden_1': 224,}