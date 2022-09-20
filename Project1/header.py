# header.py

# Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sklearn.linear_model as skl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# Functions for Vanilla Data Generation
def simple_function(x):
    return(2*x**2+6*x)

'''
****************************************************************************************
'''

# Function for Generating Design Matrix
def create_X(x,y,n):
    N = len(x)                  # Number of rows in design matrix // corr. to # of inputs to outputs
    l = int((n+1)*(n+2)/2)		# Number of elements in beta // Number of columns in design matrix // corr. to the weights
    X = np.ones((N,l))          # Initialize design matrix X

    # For simple function
    for i in range(1,n+1):   # Looping through columns 1 to n
        X[:,i] = x**(i) 

    # #"Franke"
    # for i in range(1,n+1):  # Loop through i = 1,2,3,4,5
    #     q = int((i)*(i+1)/2) # Desi
    #     for k in range(i+1):
    #         X[:,q+k]=(x**(i-k))*y**k

    return X

# Function for performing SVD on non-invertible matrix
def SVD(A): # Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD)
    U, S, VT = np.linalg.svd(A,full_matrices=True)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=S[i]
    return U @ D @ VT

# Function for returning model function
def ytilde(degree,x,fx):
    print("cock")
    X = create_X(x,0,degree)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    beta = SVD(X.T.dot(X)).dot(X.T).dot(fx)
    return X @ beta, X_train, X_test, Y_train, Y_test # Returns ytilde

# Function for calculating mean squared error (MSE)
def MSE_func(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Function for calculating relative error 
def RelativeError_func(y_data,y_model):
    return abs((y_data-y_model)/y_data)