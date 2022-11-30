# import sys
# sys.path.insert(0,"../Project1/code_and_plots/")
# from header import set_size, FrankeFunction, create_X, OLSlinreg, Ridgelinreg, MSE_func, R2
from nn import *
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from sklearn.model_selection import  train_test_split

# np.random.seed(2022)
# sns.set()

'''
Pre-process Data:
1. Split train test
2. Onehot vectorization
'''

xtrain, xtest, ytrain, ytest = train_test_split()

def onehot(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

ytrain, ytest = onehot(ytrain), onehot(ytest)