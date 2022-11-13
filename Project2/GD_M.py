# Imports
from random import random, seed
import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from sklearn.model_selection import train_test_split

from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
import autograd.numpy as np
from autograd import grad
 
# objective function
def objective(x):
	return 4.0*x**2.0 + 3.0*x + 1.0
 
# derivative of objective function
def derivative(x):
	return x * 8.0 + 3.0
#obj_grad = grad(objective)

# gradient descent algorithm - calculates gradients per step, momentum saves the data
def gd2(objective, derivative, bounds, n_iter, step_size, momentum):
	solutions, scores = list(), list()
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	change = 0.0
	print("\n")
	# run the gradient descent
	for i in range(n_iter):
		# calculate gradient <- not working since we're looking at the directory of solutions
		# and not actually a point x 
		# obj_grad = grad(objective)    
		# gradient = obj_grad(solution)
		gradient = derivative(solution)
		# calculate update
		new_change = step_size * gradient + momentum * change
		# take a step
		solution = solution - new_change
		# save the change
		change = new_change
		# evaluate candidate point
		solution_eval = objective(solution)
		# store solution
		solutions.append(solution)
		scores.append(solution_eval)
		# report progress
		print('Loop #%d grad(%s) = %.5f' % (i+1, solution, solution_eval))
	print("\n")
	return [solutions, scores]
 
# gradient descent algorithm - calculates gradients per step, momentum saves the data
def gd(objective, derivative, bounds, n_iter, step_size, momentum):
	# track all solutions
	solutions, scores = list(), list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# keep track of the change
	change = 0.0
	# run the gradient descent
	print(type(solution))
	for i in range(n_iter):
		print("gradient at ", solution)
		# calculate gradient   
		gradient = derivative(solution)
		# calculate update
		new_change = step_size * gradient + momentum * change
		# take a step
		solution = solution - new_change
		# save the change
		change = new_change
		# evaluate candidate point
		solution_eval = objective(solution)
		# store solution
		solutions.append(solution)
		scores.append(solution_eval)
		# report progress
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solutions, scores]
 
# seed the pseudo random number generator
seed(4)
# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 5
# define the step size
step_size = 0.1
# define momentum
momentum = 0.3
# perform the gradient descent search with momentum
solutions, scores = gd2(objective, derivative, bounds, n_iter, step_size, momentum)
# sample input range uniformly at 0.1 increments
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# plot the solutions found
pyplot.plot(solutions, scores, '.-', color='red')
# show the plot
# pyplot.show()