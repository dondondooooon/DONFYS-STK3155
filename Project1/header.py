# header.py

# Import Library
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as skl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import scipy as scp
import argparse
from IPython.display import display
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
# import warnings

# Vanilla 1-D Data Generation
def simple_function(x,noise,noisy):
    function_ = 7.2*x**5 + 0.5*x**2 # 2.5*np.e**-x + 0.72*x**x 
    if noisy == True:
        return function_+0.1*noise
    else:
        return function_

# Franke Function
def FrankeFunction(x,y,noise,noisy):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if noisy == True:
        return (term1+term2+term3+term4)+0.1*noise
    else:
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
def OLSlinreg(X,f):
    A = np.linalg.pinv(X.T.dot(X)) # SVD inverse
    beta = A.dot(X.T).dot(f)
    return beta # Returns optimal beta
    
# Scaling Data 
def scale_data(xtrain,xtest,ytrain,ytest):
    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrainscaled = scaler.transform(xtrain)
    xtestscaled = scaler.transform(xtest)
    scaler.fit(ytrain)
    ytrainscaled = scaler.transform(ytrain)
    ytestscaled = scaler.transform(ytest)
    return xtrainscaled, xtestscaled, ytrainscaled, ytestscaled

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

# For plotting beta as increasing order of polynomials 
def plot_beta_save(betavals,beta):
    return betavals.append(beta)

# Boostrap Resampling method
def bootstraping(xtrain,ytrain):
    N = len(xtrain)
    ind = np.random.randint(0,N,size=(N,))
    xprime = xtrain[ind]
    funcprime = ytrain[ind]
    return xprime,funcprime

# MSE and R2 via OLS
def OLS_learning(x,y,n,func,phi,noisy):
    # Initiate Containers
    degree_beta = []
    r2test = np.zeros(n)
    r2train = np.zeros(n)
    MSE_test = np.zeros(n)
    MSE_train = np.zeros(n)
    R2_sklTest = np.zeros(n)
    R2_sklTrain = np.zeros(n)
    MSE_sklTest = np.zeros(n)
    MSE_sklTrain = np.zeros(n)

    # Loop from degree 1 to max_degree polynomials
    for degree in phi: 
        X = create_X(x,y,degree) # Build Design Matrix

        # Splitting the Data
        X_train, X_test, y_train, y_test = train_test_split\
          (X,func, test_size = 0.2)#, random_state=69) 
        # # Scale the Data
        # X_train, X_test, y_train, y_test = scale_data(X_train,X_test,y_train,y_test)


        # Training 
        beta = OLSlinreg(X_train,y_train) # Calculate Beta 
        ytilde = X_train @ beta
        # plot_beta_save(degree_beta,beta) # For plotting beta as increasing order of polynomials 

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

    # For plotting beta as increasing order of polynomials    
    # BetaValues = pd.DataFrame(degree_beta)
    # BetaValues.to_csv(f'results/BetaValsDegNoise:{noisy}.csv', index=False)

    return (MSE_train,MSE_test,MSE_sklTrain,MSE_sklTest,\
        r2train,r2test,R2_sklTrain,R2_sklTrain)

# OLS MSE bootstrap resampling 
def OLS_boots(x,y,n,func,phi,nB):
    # warnings.filterwarnings('ignore') # for the bias calculation
    var = np.zeros((n,nB))
    bias = np.zeros((n,nB))
    msesamp = np.zeros((n,nB))

    for degree in phi:
        X = create_X(x,y,degree) # Build Design Matrix
        # Splitting the Data
        X_train, X_test, y_train, y_test = train_test_split\
            (X,func, test_size = 0.2)
        ypred = np.empty((y_test.shape[0],nB))
        for boots in range(0,nB):
            # Sample Data
            X_trboot, y_trboot = bootstraping(X_train,y_train)
            beta = OLSlinreg(X_trboot,y_trboot) # Calculate Beta
            ypred[:,boots] = (X_test @ beta).ravel()  # Testing
            msesamp[degree-1,boots] = MSE_func(y_test,ypred[:,boots]) # Calculate MSE
            # msesamp[degree] = np.mean( np.mean((y_test - ypred)**2, axis=1, keepdims=True) )
            bias[degree-1,boots] = np.mean( (y_test - np.mean(ypred, axis=1, keepdims=True))**2 )
            var[degree-1,boots] = np.mean( np.var(ypred, axis=1, keepdims=True) )
    # print("ferdig")
    return msesamp,bias,var 

    
    