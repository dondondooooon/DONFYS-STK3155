# header.py

# Import Library
from pickle import TRUE
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
    return 2.0+5*x*x+0.1*x

'''
****************************************************************************************
'''

# Function for Generating Design Matrix
def create_X(x,y,n,simple):
    N = len(x)                  # Number of rows in design matrix // corr. to # of inputs to outputs
    l = int((n+1)*(n+2)/2)		# Number of elements in beta // Number of columns in design matrix // corr. to the weights
    if simple == True:          # For simple function
        l = n + 1 
    X = np.ones((N,l))          # Initialize design matrix X

    for i in range(1,n+1):   # Looping through columns 1 to n
        X[:,i] = np.squeeze(x)**(i) 
        # X[i,:] = x**(i) #cuz i did rescale n .reshpae(-1,1)

    # #"Franke"
    # for i in range(1,n+1):  # Loop through i = 1,2,3,4,5
    #     q = int((i)*(i+1)/2) # Desi
    #     for k in range(i+1):
    #         X[:,q+k]=(x**(i-k))*y**k

    return X

# Function for performing SVD on non-invertible matrix
def SVD(A): # Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD)
    U, S, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=S[i]
    return U @ D @ VT

# Function for performing OLS
def mylinreg(X,fx):
    # beta = SVD(X.T.dot(X)).dot(X.T).dot(fx)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(fx)
    # np.inv() ---> try out 
    return beta # Returns optimal beta 

# Scaling Data 
def scale_data(X,y):
     scaler = StandardScaler()
     scaler.fit(X)
     Xscaled = scaler.transform(X)
     yscaled = scaler.transform(y)
     return Xscaled, yscaled

# Function for calculating mean squared error (MSE)
def MSE_func(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Function for calculating relative error 
def RelativeError_func(y_data,y_model):
    return abs((y_data-y_model)/y_data)
