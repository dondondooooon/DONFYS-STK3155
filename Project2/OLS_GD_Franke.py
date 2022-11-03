""""
import autograd.numpy as jnp
import numpy as np


In summary, you should perform an analysis of the results for OLS and Ridge
regression as function of the chosen learning rates, the number of mini-batches
and epochs as well as algorithm for scaling the learning rate. 
i.e:
    MSE vs learning rate 
    MSE vs mini batches 
    MSE vs epochs 
    MSE vs complexity for GD, GD w m, SGD, SGD w m 
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
n = 5 #max polynomial 

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


X = np.c_[np.ones((n,1)), x]
# Hessian matrix
H = (2.0/n)* X.T @ X
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
print("Beta Linreg")
print(beta_linreg)
beta = np.random.randn(2,1)

eta = 1.0/np.max(EigValues)
Niterations = 1000

for iter in range(Niterations):
    gradient = (2.0/n)*X.T @ (X @ beta-y)
    beta -= eta*gradient
print("Betas:")
print(beta)

##Print Franke function 
xmesh,ymesh = np.meshgrid(x,y)
fig = plt.figure(figsize=set_size(345), dpi=80)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xmesh, ymesh, z,\
cmap=cm.coolwarm,linewidth=0, antialiased=False) # Plot the surface.
ax.set_xlabel('x', linespacing=3.2)
ax.set_ylabel('y', linespacing=3.1)
ax.set_zlabel('z', linespacing=3.4)
fig.colorbar(surf, shrink=0.5)  # Add colorbar
# plt.savefig(f"results/FrankFunction_Noise:True.pdf", format='pdf', bbox_inches='tight')
plt.show()