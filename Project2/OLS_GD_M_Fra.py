""""
Outline for naked OLS regression with Gradient Descent 
-on the Franke Function without noise WITH momentum 
"""
# Importing various packages
from numpy.random import rand
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from numpy import asarray
from numpy import arange
from matplotlib import pyplot
from autograd import grad
 
#number of datapoints + No. of rows in design matrix // corr. to # of inputs to outputs
N = 100 
# Franke function
x = np.sort(np.random.uniform(0,1,N)) 
y = np.sort(np.random.uniform(0,1,N))
#z = np.sort(np.random.uniform(0,1,N))
term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) # First term
term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) # Second term
term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) # Third term
term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) # Fourth term
z = (term1+term2+term3+term4)
#gotta figure out why a FUNCTION for franke function fucks up the MSE'
# def franke_function():
#     term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) # First term
#     term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) # Second term
#     term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) # Third term
#     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) # Fourth term
#     z = (term1+term2+term3+term4)
#     return z

def start_func(x):
    return z #<-- need to figure out a way to get z to be a single line... maybe instead of term 1-4 we just define z in one line?

def deriv_func(x):
    return grad(z)  #<-- need to figure out a way to get z to be a single line... maybe instead of term 1-4 we just define z in one line?

# mse function 
def run_mse(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n     

# gradient descent algo w momentum 
def gradient_descent_m(start_func, deriv_func, bounds, n_iterations, step_size, momentum):
	# track all solutions
	solutions, scores = list(), list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# keep track of the change
	change = 0.0
	# run the gradient descent
	for i in range(n_iterations):
		# calculate gradient
		gradient = deriv_func(solution)
		# calculate update
		new_change = step_size * gradient + momentum * change
		# take a step
		solution = solution - new_change
		# save the change
		change = new_change
		# evaluate candidate point
		solution_eval = start_func(solution)
		# store solution
		solutions.append(solution)
		scores.append(solution_eval)
		# report progress
		print('>Gradient #%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solutions, scores]

# gradient descent loop
def gradient_loop():
    for _ in range(n_iterations):
        global beta_gradient
        gradient = (2.0/n)*X.T @ (X @ beta_gradient-z)
        #add stopping criteria here
        # if gradient < 10:
        #updating beta
        beta_gradient = beta_gradient - (gamma*gradient) #()should be -learning rate * gradient
        print("Gradient Beta #", _+1, beta_gradient)
    return beta_gradient

# gradient descent loop with beta gamma
def gradient_loop_gamma():
    for _ in range(n_iterations):
        global beta_gamma
        gradient = (2.0/n)*X.T @ (X @ beta_gamma-z)
        #add stopping criteria here
        # if gradient < 10:
        #updating beta
        beta_gamma = beta_gamma - (gamma*gradient) #()should be -learning rate * gradient
        print("Gradient Beta #", _+1, beta_gamma)
    return beta_gamma

# gradient descent loop with beta gamma AND beta_gradient -> looks like this works
def gradient_loop_gb():
    for _ in range(n_iterations):
        global beta_gamma, beta_gradient
        gradient = (2.0/n)*X.T @ (X @ beta_gamma-z)
        #add stopping criteria here
        # if gradient < 10:
        #updating beta
        beta_gamma = beta_gamma - (gamma*gradient) #()should be -learning rate * gradient
        print("Gradient Beta #", _+1, beta_gamma)
        gradient_beta = (2.0/n)*X.T @ (X @ beta_gradient-z)
        #add stopping criteria here
        # if gradient < 10:
        #updating beta
        beta_gradient = beta_gradient - (gamma*gradient_beta) #()should be -learning rate * gradient
        print("Gradient Beta #", _+1, beta_gradient)
    return beta_gamma

#defining range for input -> needs to be changed
bounds = asarray([[-1.0, 1.0]])

# initializing starter variables 
#rando seed
np.random.seed(2021) 

#max polynomial 
n = 5 
# No. of elements in beta // Number of columns in design matrix 
l = int((n+1)*(n+2)/2) 
#calling franke function 
#franke_function()

# Frank Function 2D Design Matrix X
X = np.ones((N,l))      
for i in range(1,n+1):  # Loop through features 1 to n (skipped 0)
    q = int((i)*(i+1)/2)    
    for k in range(i+1):
        X[:,q+k]=(x**(i-k))*y**k

# Hessian matrix
Hes_Mat = (2.0/n)* X.T @ X


# OLS regressionn
beta_ols = np.linalg.inv(X.T @ X) @ X.T @ z

#find out what this is !!!
beta_gradient = np.random.randn(21, )
beta_gamma = np.random.randn(21, )

# Fetch and print eigenvalues
EigValues, EigVectors = np.linalg.eig(Hes_Mat)
# print(f"Eigenvalues of Hessian Matrix:{EigValues}")


# Learning rate
lear_rate = 0.002
#gamma parameter - leraning rate
gamma = 1.0/np.max(EigValues)
n_iterations = 50

#step-size
step_size = 0.1
#momentum
momentum = 0.2
solutions, scores = gradient_descent_m(start_func, deriv_func, bounds, n_iterations, step_size, momentum)


#sample input range uniformly at 0.1 increments
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = start_func(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# plot the solutions found
pyplot.plot(solutions, scores, '.-', color='red')
# show the plot
pyplot.show()



# calling gradient_loop
# gradient_loop()
# gradient_loop_gamma()
gradient_loop_gb()

# Design matrix specifically for ypredict
ypredict_1 = X.dot(beta_gradient)
ypredict_2 = X.dot(beta_ols)
ypredict_3 = X.dot(beta_gamma)

# Running MSE on these funcs
mse_grad = run_mse(z, ypredict_1)
mse_OLS = run_mse(z, ypredict_2)
mse_gam = run_mse(z, ypredict_3)

# Printing MSE
print("Gradient MSE: ", mse_grad)
print("OLS MSE: ", mse_OLS)
print("Gamma MSE: ", mse_gam)

# Plotting functions 
# MSE vs complexity?
# MSE vs epoch? - # of times a complete dataset is passed through the algorithm  
# ^^ vs batch - small groups of a dataset is passed through the algorithm 
# kinda similar to the test train split
