import sys
sys.path.insert(1,"../../Project1/code_and_plots/")
from header import set_size, FrankeFunction, create_X, OLSlinreg, Ridgelinreg, MSE_func, R2
from mod_nn import *
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from sklearn.model_selection import  train_test_split

np.random.seed(2022)
sns.set()

# make data set and split for test and train
N = 30
degree = 5
noise = np.random.uniform(0,1,N) * 0.2
x,y,func = FrankeFunction(N,noise,False)
X = create_X(x,y,n=degree)
X_train, X_test, y_train, y_test = train_test_split(X,func, test_size = 0.2, random_state=1) 

epochs = 30
init_neurons = 10
hlayers = 3
batch_size = 5
etas = np.logspace(-5,1,10)
lambdas = etas.copy()

NN = NeuralNetwork(X_train,y_train,n_hidden_neurons=init_neurons,n_hidden_layers=hlayers,batch_size=batch_size,\
    eta=0.01,lmbd=0.0,cost='mse',activation='sigmoid',score='mse',output_activation=None,xtest=X_test,ytest=y_test)
NN.SGD_train(1000)
ypred = NN.predict(X_test)
print("\n\nPrediction:", ypred.ravel())
print("\nTarget:",y_test)
print("\nMSE:",MSE_func(y_test,ypred))

plt.plot(NN.escore,label="Train")
plt.plot(NN.testscore,label="Test")
plt.legend()
plt.show()

'''
--------
'''
# '''
# OLS tuning
# '''
# # OLS
# betaols = OLSlinreg(X_train,y_train,0)
# ypred_ols = X_test @ betaols

# # params
# epochs = 500
# init_neurons = 10
# hlayers = 1
# batch_size = 5
# etas = np.logspace(-5,1,10)
# lambdas = etas.copy()
# gammas = etas.copy()

# # # given eta = 0.01; MSE vs. Epochs for different gamma
# # mse_epoch_gamma = np.zeros((len(gammas), epochs + 1))
# # itera = 0
# # for gamma in gammas:
# #     NN = NeuralNetwork(X_train,y_train,n_hidden_neurons=neurons,n_hidden_layers=hlayers,batch_size=batch_size,\
# #         eta=0.01,lmbd=0.0,gamma=gamma,cost="mse",activation="sigmoid",score="mse",output_activation=None)
# #     NN.SGD_train(epochs)
# #     mse_epoch_gamma[itera,:] = NN.escore.ravel() 
# #     itera += 1

# # Concluded gammas[5] works best (≈ 4.6e-03) for eta = 0.01
# # and that only needs about less than 30 epochs,
# # but will still use 100 to be sure

# # # eta gamma heatmap tuning
# epochs = 100
# # mse_eta_gamma = np.zeros((len(etas),len(gammas)))
# # itera = 0
# # for eta in etas:
# #     jtera = 0
# #     for gamma in gammas:
# #         NN = NeuralNetwork(X_train,y_train,n_hidden_neurons=neurons,n_hidden_layers=hlayers,batch_size=batch_size,\
# #             eta=eta,lmbd=0.0,gamma=gamma,cost="mse",activation="sigmoid",score="mse",output_activation=None)
# #         # train network
# #         NN.SGD_train(epochs)
# #         ypred = NN.predict(X_test)
# #         mse_eta_gamma[itera,jtera] = MSE_func(y_test,ypred)
# #         jtera += 1
# #     itera += 1

# # concluded gammas[1] and etas[3] give the best MSE_test

# # # network architecture tuning
# # hidden_layers = np.arange(1,21,1) # y
# # neuron_range = np.arange(1,101,1) # x
# # mse_arch = np.zeros((20,100))
# # itera = 0
# # for hl in hidden_layers:
# #     jtera = 0
# #     for neu in neuron_range:
# #         NN = NeuralNetwork(X_train,y_train,n_hidden_neurons=neu,n_hidden_layers=hl,batch_size=batch_size,\
# #             eta=etas[3],lmbd=0.0,gamma=gammas[1],cost="mse",activation="sigmoid",score="mse",output_activation=None)
# #         # train network
# #         NN.SGD_train(epochs)
# #         ypred = NN.predict(X_test)
# #         mse_arch[itera,jtera] = MSE_func(y_test,ypred)
# #         jtera += 1
# #     itera += 1

# # concluded that 1 hidden layer and 86 neurons
# # gave best MSE of 0.018



# '''
# Ridge tuning
# '''
# # # eta x lambda tuning
# # mse_eta_lmb = np.zeros((len(etas),len(lambdas)))
# # itera = 0
# # for eta in etas:
# #     jtera = 0
# #     for lmb in lambdas:
# #         NN = NeuralNetwork(X_train,y_train,n_hidden_neurons=init_neurons,n_hidden_layers=1,batch_size=batch_size,\
# #             eta=eta,lmbd=lmb,gamma=gammas[1],cost="mse",activation="sigmoid",score="mse",output_activation=None)
# #         # train network
# #         NN.SGD_train(epochs)
# #         ypred = NN.predict(X_test)
# #         mse_eta_lmb[itera,jtera] = MSE_func(y_test,ypred)
# #         jtera += 1
# #     itera += 1

# # concluded that eta[3] = 10^-3.0 and lambdas[1] = 10^-4.33
# # gave the optimal test MSE score of 0.044

# # # network architecture tuning
# # hidden_layers = np.arange(1,21,1) # y
# # neuron_range = np.arange(1,101,1) # x
# # ridge_mse_arch = np.zeros((20,100))
# # itera = 0
# # for hl in hidden_layers:
# #     jtera = 0
# #     for neu in neuron_range:
# #         NN = NeuralNetwork(X_train,y_train,n_hidden_neurons=neu,n_hidden_layers=hl,batch_size=batch_size,\
# #             eta=etas[3],lmbd=lambdas[1],gamma=gammas[1],cost="mse",activation="sigmoid",score="mse",output_activation=None)
# #         # train network
# #         NN.SGD_train(epochs)
# #         ypred = NN.predict(X_test)
# #         ridge_mse_arch[itera,jtera] = MSE_func(y_test,ypred)
# #         jtera += 1
# #     itera += 1

# # concluded optimal hidden layers = 1 w/ 86 neurons giving MSE of 0.018

# # Ridge
# betaridge = Ridgelinreg(X_train,y_train,lambdas[1])
# ypred_ridge = X_test @ betaridge

# '''
# Saving Matrices
# '''
# # np.savetxt(f'saved_txt/epoch_gamma.txt', mse_epoch_gamma)
# # np.savetxt(f'saved_txt/ols/MSE_eta_gamma.txt', mse_eta_gamma)
# # np.savetxt(f'saved_txt/ols/network_architecture.txt', mse_arch)
# # np.savetxt(f'saved_txt/ridge/MSE_eta_lmb.txt', mse_eta_lmb)
# # np.savetxt(f'saved_txt/ridge/network_architecture.txt',ridge_mse_arch)


# '''
# Final Comparison
# '''
# epochs = 300
# # optimal OLS
# nnols = NeuralNetwork(X_train,y_train,n_hidden_neurons=86,n_hidden_layers=1,batch_size=batch_size,\
#     eta=etas[3],lmbd=0.0,gamma=gammas[1],cost="mse",activation="sigmoid",score="mse",output_activation=None)
# # train network
# nnols.SGD_train(epochs)
# ypred_nn_ols = nnols.predict(X_test)

# # optimal Ridge
# nnridge = NeuralNetwork(X_train,y_train,n_hidden_neurons=86,n_hidden_layers=1,batch_size=batch_size,\
#     eta=etas[3],lmbd=lambdas[1],gamma=gammas[1],cost="mse",activation="sigmoid",score="mse",output_activation=None)
# # train network
# nnridge.SGD_train(epochs)
# ypred_nn_ridge = nnridge.predict(X_test)


# ols_mse = MSE_func(y_test,ypred_ols)
# ols_nn_mse = MSE_func(y_test,ypred_nn_ols)
# ridge_mse = MSE_func(y_test,ypred_ridge)
# ridge_nn_mse = MSE_func(y_test,ypred_nn_ridge)

# ols_r2 = R2(y_test,ypred_ols)
# ols_nn_r2 = R2(y_test,ypred_nn_ols)
# ridge_r2 = R2(y_test,ypred_ridge)
# ridge_nn_r2 = R2(y_test,ypred_nn_ridge)


# print("OLS:")
# print("prev: MSE = ",ols_mse, ", R2 = ", ols_r2)
# print("nn: MSE = ",ols_nn_mse, ", R2 = ", ols_nn_r2)

# print("RIDGE:")
# print("prev: MSE = ",ridge_mse, ", R2 = ", ridge_r2)
# print("nn: MSE = ",ridge_nn_mse, ", R2 = ", ridge_nn_r2)
