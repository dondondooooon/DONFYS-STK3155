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
import argparse
from IPython.display import display

# Vanilla 1-D Data Generation
def simple_function(x,noise,noisy):
    function_ = 7.2*x**5 + 0.5*x**2 # 2.5*np.e**-x + 0.72*x**x 
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
def create_X(x,y,n):
    N = len(x)              # No. of rows in design matrix // corr. to # of inputs to outputs
    l = int((n+1)*(n+2)/2)  # No. of elements in beta // Number of columns in design matrix // corr. to the weights
    # Frank Function 2D Design Matrix
    X = np.ones((N,l))      # Initialize design matrix X
    for i in range(1,n+1):  # Loop through features 1 to n (skipped 0)
        q = int((i)*(i+1)/2)    
        for k in range(i+1):
            X[:,q+k]=(x**(i-k))*y**k  # Calculate the right polynomial term
    # # For simple 1D Function Design Matrix
    # l = n + 1 
    # X = np.ones((N,l))      # Initialize design matrix X
    # for i in range(1,n+1):  # Looping through columns 1 to n
    #     X[:,i] = x**i #np.squeeze?
    return X

'''
****************************************************************************************
'''

# Function for performing OLS
def mylinreg(X,fx):
    A = np.linalg.pinv(X.T.dot(X)) # SVD inverse
    beta = A.dot(X.T).dot(fx)
    return beta # Returns optimal beta
    
# Scaling Data 
def scale_data(xtrain,xtest):
    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrainscaled = scaler.transform(xtrain)
    xtestscaled = scaler.transform(xtest)
    return xtrainscaled, xtestscaled

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

# MSE and R2 as functions of Complexity 
def complexity_dependencies(x,y,n,func,phi):
    MSE_sklTrain = np.zeros(n)
    MSE_sklTest = np.zeros(n)
    R2_sklTrain = np.zeros(n)
    R2_sklTest = np.zeros(n)
    MSE_train = np.zeros(n)
    MSE_test = np.zeros(n)
    r2train = np.zeros(n)
    r2test = np.zeros(n)

    for degree in phi: # skipped 0th complexity 
        X = create_X(x,y,degree) # Build Design Matrix
        # Splitting the Data
        X_train, X_test, y_train, y_test = train_test_split\
          (X,func, test_size = 0.2)#, random_state=69) 
        # # Scale the Data
        # X_train, X_test = scale_data(X_train,X_test)
        # Training 
        beta = mylinreg(X_train,y_train) # Beta 
        ytilde = X_train @ beta # Model Function
        # Testing
        ypredict = X_test @ beta
        # MSE & R2 score via own Algorithm
        MSE_train[degree-1] = MSE_func(y_train,ytilde)  
        MSE_test[degree-1] =  MSE_func(y_test,ypredict) 
        r2train[degree-1] = R2(y_train,ytilde)
        r2test[degree-1] = R2(y_test,ypredict)
        # SciKitLearnRegCheck
        clf = skl.LinearRegression().fit(X_train,y_train) # fit_intercept=False ?
        MSE_sklTrain[degree-1] = mean_squared_error(clf.predict(X_train),y_train)
        MSE_sklTest[degree-1] = mean_squared_error(clf.predict(X_test),y_test)
        R2_sklTrain[degree-1] = clf.score(X_train,y_train)
        R2_sklTest[degree-1] = clf.score(X_test,y_test)
        # Display BetaValues... maybe export and then replot them? 
        BetaValues = pd.DataFrame(beta)
        BetaValues.columns = [r'$\beta$']
        display(BetaValues)

    return (MSE_train,MSE_test,MSE_sklTrain,MSE_sklTest,\
        r2train,r2test,R2_sklTrain,R2_sklTrain)