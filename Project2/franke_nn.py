#from neuralnetwork import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
np.random.seed(0)

# Franke Function
def FrankeFunction(N,noise):
     x = np.sort(np.random.uniform(0,1,N)) 
     y = np.sort(np.random.uniform(0,1,N))
     term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) # First term
     term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) # Second term
     term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) # Third term
     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) # Fourth term
     z = (term1+term2+term3+term4)+noise
     return x,y,z # Final function plus eventual noise 

# Generating Design Matrix
def create_X(x,y,n):
    N = len(x)              # No. of rows in design matrix // corr. to # of inputs to outputs
    l = int((n+1)*(n+2)/2)  # No. of elements in beta // Number of columns in design matrix 
    # Frank Function 2D Design Matrix
    X = np.ones((N,l))      # Initialize design matrix X
    for i in range(1,n+1):  # Loop through features 1 to n (skipped 0)
        q = int((i)*(i+1)/2)    
        for k in range(i+1):
            X[:,q+k]=(x**(i-k))*y**k  # Calculate the right polynomial term
    return X

# Function for performing OLS regression
def OLSlinreg(X,f):
    A = np.linalg.pinv(X.T.dot(X)) # SVD inverse
    beta = A.dot(X.T).dot(f)
    return beta # Returns optimal beta

# Mean Squared Error (MSE)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n 


N = 30
noise = 0
x,y,func = FrankeFunction(N,noise)
X = create_X(x,y,5)
X_train, X_test, y_train, y_test = train_test_split(X,func, test_size = 0.2, random_state=1) 

beta = OLSlinreg(X_train,y_train)
ytilde_ols = X_test @ beta

print(func.shape,ytilde_ols.shape)

# NN = NeuralNetwork(X_train,y_train,n_hidden_neurons=10,n_categories=10,epochs=10,batch_size=5,eta=0.1,lmbd=0.0)
# NN.SGD_train()
# ind,predict,que = NN.predict(X_test)
# print(predict.shape)
# # print(y_test.shape)
# print(predict)
# print(y_test)
# print(que.shape)

# print(ind)
