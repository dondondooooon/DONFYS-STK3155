import sys
sys.path.insert(0,"../Project1/code_and_plots/")
from header import set_size
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick

# Load chunky matrices
eg = np.loadtxt(f'saved_txt/epoch_gamma.txt')
nnmse = np.loadtxt(f'saved_txt/ols/MSE_eta_gamma.txt')
arch = np.loadtxt(f'saved_txt/ols/network_architecture.txt')
mselmbeta = np.loadtxt(f'saved_txt/ridge/MSE_eta_lmb.txt')
ridge_arch = np.loadtxt(f'saved_txt/ridge/network_architecture.txt')


# processing
gamma_len, epoch_len = eg.shape
epochs = np.arange(0,501,1)
gammas = np.logspace(-5,1,gamma_len)
g = r"$\gamma$"
eta_len, gamma_len = nnmse.shape
etas = np.logspace(-5,1,eta_len)
layers, neur = arch.shape
hidden_layers = np.arange(1,layers+1,1) 
neuron_range = np.arange(1,neur+1,1) 
eta_len, lambda_len = mselmbeta.shape
lambdas = np.logspace(-5,1,lambda_len)


# # gamma vs epoch
# fig, ax = plt.subplots(figsize=set_size(345), dpi=80)
# for i in range(gamma_len):
#     plt.plot(eg[i,:125],label=f"{g} = {gammas[i]:.1e}")
# ax.set_xlabel("epoch")
# ax.set_ylabel(r"MSE$_{train}$")
# plt.legend()
# # plt.savefig(f'results/epoch_gamma.pdf', format='pdf', bbox_inches='tight')
# plt.show()


# # eta x gamma matrix: OLS 
# xtix, ytix = np.around(np.log10(gammas),2), np.around(np.log10(etas),2)
# fig, ax = plt.subplots(figsize=set_size(345), dpi=80)
# sns.heatmap(nnmse, xticklabels=xtix, yticklabels=ytix, annot=True, annot_kws={"fontsize":7},\
#     ax=ax, cmap=cm.coolwarm,cbar_kws={'label': r'MSE$_{test}$'})
# ax.set_xlabel(r"log$\gamma$")
# ax.set_ylabel(r"log$\eta$")
# # plt.savefig(f'results/eta_gamma.pdf', format='pdf', bbox_inches='tight')
# plt.show()


# # hidden layer and neurons heatmap
# i = 0
# for j in range(1,11):
#     start, end = 10*i+1, 10*j
#     if i == 0:
#         start = 0
#     fig, ax = plt.subplots(figsize=set_size(345), dpi=80)
#     xtix, ytix = neuron_range[start:end], hidden_layers
#     sns.heatmap(arch[ : , start:end], xticklabels=xtix, yticklabels=ytix, annot=True, annot_kws={"fontsize":7},\
#         ax=ax, cmap=cm.coolwarm,cbar_kws={'label': r'MSE$_{test}$'})
#     ax.set_xlabel(r"Number of Neurons")
#     ax.set_ylabel(r"Number of Hidden Layers")
#     plt.savefig(f'results/neu_archs/ols/heatmap{start}_{end}.pdf', format='pdf', bbox_inches='tight')
#     plt.show()
#     i += 1
# ind = np.unravel_index(np.argmin(arch, axis=None), arch.shape)
# print("index:", ind)
# print("val:", arch[ind])


# # eta x lambda heatmap ridge
# xtix, ytix = np.around(np.log10(lambdas),2), np.around(np.log10(etas),2)
# fig, ax = plt.subplots(figsize=set_size(345), dpi=80)
# sns.heatmap(mselmbeta, xticklabels=xtix, yticklabels=ytix, annot=True, annot_kws={"fontsize":7},\
#     ax=ax, cmap=cm.coolwarm,cbar_kws={'label': r'MSE$_{test}$'})
# ax.set_xlabel(r"log$\lambda$")
# ax.set_ylabel(r"log$\eta$")
# # plt.savefig(f'results/eta_lmb_ridge.pdf', format='pdf', bbox_inches='tight')
# plt.show()


# # # RIDGE hidden layer and neurons heatmap
# i = 0
# for j in range(1,11):
#     start, end = 10*i+1, 10*j
#     if i == 0:
#         start = 0
#     fig, ax = plt.subplots(figsize=set_size(345), dpi=80)
#     xtix, ytix = neuron_range[start:end], hidden_layers
#     sns.heatmap(ridge_arch[ : , start:end], xticklabels=xtix, yticklabels=ytix, annot=True, annot_kws={"fontsize":7},\
#         ax=ax, cmap=cm.coolwarm,cbar_kws={'label': r'MSE$_{test}$'})
#     ax.set_xlabel(r"Number of Neurons")
#     ax.set_ylabel(r"Number of Hidden Layers")
#     plt.savefig(f'results/neu_archs/ridge/heatmap{start}_{end}.pdf', format='pdf', bbox_inches='tight')
#     plt.show()
#     i += 1
# ind = np.unravel_index(np.argmin(arch, axis=None), arch.shape)
# print("index:", ind)
# print("val:", arch[ind])