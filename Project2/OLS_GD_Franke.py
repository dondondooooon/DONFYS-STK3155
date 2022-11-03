""""
import autograd.numpy as jnp
import numpy as np


In summary, you should perform an analysis of the results for OLS and Ridge
regression as function of the chosen learning rates, the number of mini-batches
and epochs as well as algorithm for scaling the learning rate. 
i.e:
    MSE vs different learning rates vs complexity (OLS)
    PLOTS
    MSE vs mini batches 
    MSE vs epochs with different learning rates
    MSE vs complexity for GD, GD w m, SGD, SGD w m for Franke 

    .PY 
    GD_Franke: Code for GD on Franke function w/o momentum and fixed learning rate
    GDM_Franke: Code for GD on Franke function w momentum and fixed learning rate - compare to above
    SGD_Franke: Code for SGD on Franke function w/o momentum, fixed learning rate, w epochs, mini batches 
    SGDM_Franke: Code for SGD on Franke function w momentum, fixed learning rate, w epochs, mini batches
    AG_GD_Franke:  --"-- with autograd
    AG_GDM_Franke:  --"-- with autograd
    AG_SDG_Franke:  --"-- with autograd
    AG_SDGM_Franke:  --"-- with autograd
    add RMSProp and ADAM


#random seed 
np.random.seed(2021)
#meshgrid of datapoints
#test & train split 
#noise?
#polynomial degree
"""
# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
#initializing starter variables 
np.random.seed(2021) #rando seed
N = 100 #number of datapoints 
n = 2 #max polynomial 

#Franke function
x = np.sort(np.random.uniform(0,1,N)) 
y = np.sort(np.random.uniform(0,1,N))
term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) # First term
term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) # Second term
term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) # Third term
term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) # Fourth term
z = (term1+term2+term3+term4)


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

#mse
def mseFunc(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n     


#Design matrix 
#N = len(x)     not needed         # No. of rows in design matrix // corr. to # of inputs to outputs
l = int((n+1)*(n+2)/2)  # No. of elements in beta // Number of columns in design matrix 
# Frank Function 2D Design Matrix
# print("l", l)
X = np.ones((N,l))      # Initialize design matrix X
for i in range(1,n+1):  # Loop through features 1 to n (skipped 0)
    q = int((i)*(i+1)/2)    
    for k in range(i+1):
        X[:,q+k]=(x**(i-k))*y**k

print("X shape", X.shape)

# Hessian matrix
Hes_Mat = (2.0/n)* X.T @ X

# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(Hes_Mat)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

#OLS regressionn
beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ z

# #Beta Linreg values
print("Beta Linreg Shape", beta_linreg.shape)

#find out what this is 
beta = np.random.randn(6, )
# print("X?")
# print(X.shape)
print("Betas shape", beta.shape)

# print("z")
# print(z)
nu_beta = np.random.randn(6, )
lear_rate = 0.002
#gamma parameter - leraning rate
eta = 1.0/np.max(EigValues)
N_iterations = 1000

#gradient descent 
for iter in range(N_iterations):
    gradient = (2.0/n)*X.T @ (X @ beta-z)
    nu_gradient = (2.0/n)*X.T @ (X @ nu_beta-z)
    #add stopping criteria here
    # if gradient < 10:
    #updating beta
    beta = beta - (eta*gradient) #()should be -learning rate * gradient
    nu_beta = nu_beta - (lear_rate*nu_gradient)


print("Betas shape v2", beta.shape)
#design matrix specifically for ypredict
# xnew = np.array([[0],[2]])
# xbnew = np.c_[np.ones((2,1)), xnew]
ypredict = X.dot(beta)
ypredict_2 = X.dot(beta_linreg)
ypredict_3 = X.dot(nu_beta)

print("ypredict shape", ypredict.shape)
mse_grad = mseFunc(z, ypredict)
mse_OLS = mseFunc(z, ypredict_2)
mse_gam = mseFunc(z, ypredict_3)

print("MSE Grad: ", mse_grad)
print("MSE OLS: ", mse_OLS)
print("MSE Gam: ", mse_gam)

#array of gammas -> grid search 
