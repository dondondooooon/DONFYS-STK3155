# header.py

# Import Library
from pickletools import read_uint1
import sys
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
import scipy as scp

# Command Line Error
def commandline_check():
    if len(sys.argv) <= 2:
        print('Command Line Error: Check your command line arguments')
        exit(1)

# Vanilla 1-D Data Generation
def simple_function(x,noise,noisy):
    function_ = np.e**-x + 0.5*x**3 # 2.0+5*x*x
    if noisy == True:
        return function_+0.1*noise
    else:
        return function_

# Franke Function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Generating Design Matrix
def create_X(x,y,n,simple):
    N = len(x)                  # Number of rows in design matrix // corr. to # of inputs to outputs
    l = int((n+1)*(n+2)/2)		# Number of elements in beta // Number of columns in design matrix // corr. to the weights
    if simple == True:          # For simple 1D function
        l = n + 1 
        X = np.ones((N,l))      # Initialize design matrix X
        for i in range(1,n+1):  # Looping through columns 1 to n
            X[:,i] = x**i #np.squeeze?

    else: # Frank Function Design Matrix
        X = np.ones((N,l))          # Initialize design matrix X
        for i in range(1,n+1):      # Loop through features 1 to n (skipped 0)
            q = int((i)*(i+1)/2)    
            for k in range(i+1):
                X[:,q+k]=(x**(i-k))*y**k  # Calculate the right polynomial term
    return X

'''
****************************************************************************************
'''

# Function for performing SVD on non-invertible matrix
def SVD(A): # Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD)
    U, S, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=S[i]
    return U @ D @ VT

# Function for performing OLS
def mylinreg(X,fx):
    A = X.T.dot(X)
    # beta = SVD(A).dot(X.T).dot(fx)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(fx)
    # beta = np.linalg.pinv(A).dot(X.T).dot(fx)
    # np.pinv does the same thing try it out!!!
    return beta # Returns optimal beta
    
# Scaling Data 
def scale_data(X,y):
     scaler = StandardScaler()
     scaler.fit(X)
     Xscaled = scaler.transform(X)
     yscaled = scaler.transform(y)
     return Xscaled, yscaled

# Mean Squared Error (MSE)
def MSE_func(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n 

# R2 Score
def R2(y_data, y_model):
    return 1-np.sum( (y_data - y_model)**2) / np.sum( (y_data - np.mean(y_data)) ** 2 )

# Relative error 
def RelativeError_func(y_data,y_model):
    return abs((y_data-y_model)/y_data)

# Plot the actual function
def function_show(x,func):
    plt.plot(x,func)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("The Function")
    plt.show() 