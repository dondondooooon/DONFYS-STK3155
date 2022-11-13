# Imports
from random import random, seed
import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from sklearn.model_selection import train_test_split

# the number of datapoints
n = 100
x = np.random.rand(n,1)
y = 3.0*x**2 - 3.0*x + 1 + 0.1*np.random.randn(n,1)

# 2nd order polynomial in func form 
def start_f():
    return 3.0*x**2 - 3.0*x + 1 + 0.1*np.random.randn(n,1)

# derived 2nd order polynomial in func form 
def der_f():
    return 3.0*x**2 - 3.0*x + 1 + 0.1*np.random.randn(n,1)

# mse func
def run_mse(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n  
#R2 func
def run_r2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

# Design matrix 
X = np.c_[np.ones((n,1)), x]
XT_X = X.T @ X
# Hessian matrix
H = (2.0/n)* X.T @ X
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
#print(f"Eigenvalues of Hessian Matrix:{EigValues}")

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# OLS regression 
beta_linreg = np.linalg.inv(XT_X) @ (X.T @ y)
#printing  beta of regression
#print(beta_linreg)
beta = np.random.randn(2,1)

# ETA IS INVERSE OF HES MATRIX? POSSIBLY 
eta = 1.0/np.max(EigValues)
N_iterations = 1000

# Gradient descent loop
for iter in range(N_iterations):
    gradient = (2.0/n)*X.T @ (X @ beta-y)
    beta -= eta*gradient
    # print to see betas per iter, remember to reduce iterations 
    #print(iter, gradient[0], gradient[1])
    
# Printing beta after gd loop
# print(beta)

# Predicting mse, r2 and relative error for training data
ytilde = X_train @ beta
ypredict = X_test @ beta

print("\n--Training data eval--")
print("R2: ", run_r2(y_train,ytilde))
print("MSE: ", run_mse(y_train,ytilde))

print("--Test data eval--")
print("R2: ", run_r2(y_test,ypredict))
print("MSE: ", run_mse(y_test,ypredict), "\n")
