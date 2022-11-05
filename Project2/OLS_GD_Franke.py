""""
Outline for naked OLS regression with Gradient Descent 
-on the Franke Function without noise
1. Imports 
2. Initialize Franke function
3. Set size()
4. runMSE()
5. Seed & Initialize variables 
6. Setting up Design matrix
7. Hessian matrix
8. OLS regression
9. Fetch and print eigenvalues
10. Learning rate
11. Gradient Descent loop
12. Design matrix specifically for ypredict
13. Printing shape of ypredicts
14. Running MSE on these funcs
15. Printing MSE for GD, OLS, LR... ask Don abt this 
"""
# 1. Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

# 2. initializing starter variables 
#rando seed
np.random.seed(2021) 
#number of datapoints + No. of rows in design matrix // corr. to # of inputs to outputs
N = 100 
#max polynomial 
n = 5 
# No. of elements in beta // Number of columns in design matrix 
l = int((n+1)*(n+2)/2)  

# 3. Franke function
x = np.sort(np.random.uniform(0,1,N)) 
y = np.sort(np.random.uniform(0,1,N))
term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) # First term
term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) # Second term
term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) # Third term
term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) # Fourth term
z = (term1+term2+term3+term4)


# 4. set size of plots - not needed yet 
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


# 5. mse
def run_mse(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n     





# 6. Frank Function 2D Design Matrix X
X = np.ones((N,l))      
for i in range(1,n+1):  # Loop through features 1 to n (skipped 0)
    q = int((i)*(i+1)/2)    
    for k in range(i+1):
        X[:,q+k]=(x**(i-k))*y**k

print("X shape", X.shape)

# 7. Hessian matrix
Hes_Mat = (2.0/n)* X.T @ X


#8. OLS regressionn
beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ z

#find out what this is !!!
beta = np.random.randn(21, )
nu_beta = np.random.randn(21, )


#9. Fetch and print eigenvalues
EigValues, EigVectors = np.linalg.eig(Hes_Mat)
# print(f"Eigenvalues of Hessian Matrix:{EigValues}")


# 10. Learning rate
lear_rate = 0.002
#gamma parameter - leraning rate
gamma = 1.0/np.max(EigValues)
N_iterations = 1000


#11. gradient descent loop
for iter in range(N_iterations):
    gradient = (2.0/n)*X.T @ (X @ beta-z)
    nu_gradient = (2.0/n)*X.T @ (X @ nu_beta-z)
    #add stopping criteria here
    # if gradient < 10:
    #updating beta
    beta = beta - (gamma*gradient) #()should be -learning rate * gradient
    nu_beta = nu_beta - (lear_rate*nu_gradient)



# 12. Design matrix specifically for ypredict
ypredict_1 = X.dot(beta)
ypredict_2 = X.dot(beta_linreg)
ypredict_3 = X.dot(nu_beta)

# 13. Printing shape of ypredicts
print("ypredict1 shape", ypredict_1.shape)
print("ypredict2 shape", ypredict_2.shape)
print("ypredict3 shape", ypredict_3.shape)

# 14. Running MSE on these funcs
mse_grad = run_mse(z, ypredict_1)
mse_OLS = run_mse(z, ypredict_2)
mse_gam = run_mse(z, ypredict_3)

# 15. Printing MSE
print("MSE Grad: ", mse_grad)
print("MSE OLS: ", mse_OLS)
print("MSE Gam: ", mse_gam)

# 16. Plotting functions 
# MSE vs complexity?
# MSE vs epoch? - # of times a complete dataset is passed through the algorithm  
# ^^ vs batch - small groups of a dataset is passed through the algorithm 
# kinda similar to the test train split


#Expected result for 5 polynomials 
# MSE Grad:  0.015655769782326717
# MSE OLS:  0.0016947104282061534
# MSE Gam:  0.01764043437456315


# Print functions
#Print Design matrix
# print("X?")
# print(X.shape)
#Print Beta Linreg values
# print("Beta Linreg Shape", beta_linreg.shape)
#Print Betas shape
# print("Betas shape", beta.shape) - before loop
# print("Betas shape v2", beta.shape) - after loop to see the diff
#Print whatever the fuck z is
# print("z")
# print(z)

#Extra
# xnew = np.array([[0],[2]]) - idk wtf this is im ngl
# xbnew = np.c_[np.ones((2,1)), xnew] - idk wtf this is im ngl
#array of gammas -> grid search 