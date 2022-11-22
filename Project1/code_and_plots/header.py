# Import Library
import sys
import argparse
import warnings
import numpy as np
import scipy as scp
import pandas as pd
from matplotlib import cm
from imageio import imread
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import linear_model
warnings.filterwarnings('ignore')
import sklearn.linear_model as skl
from IPython.display import display
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import  train_test_split
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

np.random.seed(2018) # Setting Random Seed constant

# Set figure dimensions to avoid scaling in LaTeX.
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

# Franke Function
def FrankeFunction(N,noise,print):
    x = np.sort(np.random.uniform(0,1,N)) 
    y = np.sort(np.random.uniform(0,1,N))
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) # First term
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) # Second term
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) # Third term
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) # Fourth term
    z = (term1+term2+term3+term4)+noise

    # For printing plot
    if print == True:
        xmesh,ymesh = np.meshgrid(x,y)
        fig = plt.figure(figsize=set_size(345), dpi=80)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xmesh, ymesh, z,\
            cmap=cm.coolwarm,linewidth=0, antialiased=False) # Plot the surface.
        ax.set_xlabel('x', linespacing=3.2)
        ax.set_ylabel('y', linespacing=3.1)
        ax.set_zlabel('z', linespacing=3.4)
        fig.colorbar(surf, shrink=0.5)  # Add colorbar
        # plt.savefig(f"results/FrankFunction_Noise-True.pdf", format='pdf', bbox_inches='tight')
        plt.show()

    return x,y,z # Final function plus eventual noise

# Function for Terrain Data
def RealFunction(N,noise,print):
    terrain = imread('SRTM_data_Norway_1.tif')[::N,::N] # Load the terrain (scaled down)
    xlen,ylen = np.shape(terrain) # Get the shape (not necessarily square image)
    x = np.linspace(0,xlen-1,xlen)
    y = np.linspace(0,ylen-1,ylen)
    xmesh, ymesh = np.meshgrid(x, y, indexing='ij')
    x_flat = xmesh.reshape(xmesh.shape[0] * xmesh.shape[1])  # flattens x
    y_flat = ymesh.reshape(ymesh.shape[0] * ymesh.shape[1])  # flattens y
    z_flat = terrain.reshape(terrain.shape[0]*terrain.shape[1]) # flattens z 
    
    # For printing plot
    if print == True:
        fig = plt.figure(figsize=set_size(345),dpi=80)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xmesh,ymesh,terrain,cmap=cm.coolwarm,linewidth=0,antialiased=False)
        # ax.view_init(elev=90, azim=0)
        fig.colorbar(surf,shrink=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # plt.savefig(f'results/TerrainCMtopview.pdf', format='pdf', bbox_inches='tight')
        # plt.savefig(f'results/TerrainCM.pdf', format='pdf', bbox_inches='tight')
        plt.show()
    
    return x_flat,y_flat,z_flat # Returns flattened arrays of x,y, and z (the image)

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
    # # For simple 1D Function Design Matrix
    # l = n + 1 
    # X = np.ones((N,l))      # Initialize design matrix X
    # for i in range(1,n+1):  # Looping through columns 1 to n
    #     X[:,i] = x**i

'''
---****************************---
The following are functions for 
finding optimal beta depending 
on the regression method
'''

# Function for performing OLS regression
def OLSlinreg(X,f,lmb):
    A = np.linalg.pinv(X.T.dot(X)) # SVD inverse
    beta = A.dot(X.T).dot(f)
    return beta # Returns optimal beta

# Function for performing Ridge regression given a lambda
def Ridgelinreg(X,f,lmb):
    I = np.eye(X.shape[1])
    A = np.linalg.pinv(X.T @ X + lmb*I) # SVD inverse
    beta = A @ X.T @ f
    return beta # Returns optimal beta 

# Functin for performing Lasso regression given a lambda 
def Lassolinreg(X,f,lmb): 
    RegLasso = linear_model.Lasso(lmb) # fit_intercept=false? to remove intercept
    RegLasso.fit(X,f)
    return RegLasso.coef_

'''
---****************************---
'''

# Mean Squared Error (MSE)
def MSE_func(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data.ravel()-y_model.ravel())**2)/n 

# R2 Score
def R2(y_data, y_model):
    return 1-np.sum( (y_data - y_model.ravel())**2) / np.sum( (y_data - np.mean(y_data)) ** 2 )

# No Data Scale
def noscale(xtrain,xtest,ytrain,ytest):
    return xtrain,xtest,ytrain,ytest

# Scaling Data 
def scale_data(xtrain,xtest,ytrain,ytest):
    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrainscaled = scaler.transform(xtrain)
    xtestscaled = scaler.transform(xtest)
    scaler.fit(ytrain.reshape(-1, 1))
    ytrainscaled = scaler.transform(ytrain.reshape(-1, 1)).ravel()
    ytestscaled = scaler.transform(ytest.reshape(-1, 1)).ravel()
    return xtrainscaled, xtestscaled, ytrainscaled, ytestscaled

# Boostrap resampling
def bootstrap(xtrain,ytrain):
    N = len(xtrain)
    ind = np.random.randint(0,N,size=(N,))
    xprime = xtrain[ind]
    funcprime = ytrain[ind]
    return xprime,funcprime

# For plotting beta as increasing order of polynomials 
def plot_beta_save(betavals,beta):
    return betavals.append(beta)

'''
The following functions are for plotting and printing
'''

# Plot MSE and R2 as function of complexity + print MSE info
def ols_first(MSE_train,MSE_test,MSE_sklTrain,MSE_sklTest,r2train,\
    r2test,R2_sklTrain,R2_sklTest,phi,printed,sklcompare,title):
    width = 345
    if printed == True: # Print Facts
        print("\nThe complexity with the min. MSE in training:",\
            phi[np.argmin(MSE_train)])
        print("The complexity with the min. MSE in test:",\
            phi[np.argmin(MSE_test)], "\n")
        print("MSE_TRAIN: ", MSE_train)
        print("MSE_TEST: ", MSE_test, "\n")
        print("R2_TRAIN: ", r2train)
        print("R2_TEST: ", r2test, "\n")
        if sklcompare == True:
            print("Algo. MSETrain Diff.: ", MSE_sklTrain-MSE_train)
            print("Algo. MSETest Diff.: ", MSE_sklTest-MSE_test, "\n")
            print("Algo. R2Train Diff.: ", R2_sklTrain-r2train)
            print("Algo. R2Test Diff.: ", R2_sklTest-r2test, "\n")

    # MSE plot
    plt.style.use("ggplot") 
    plt.figure(figsize=set_size(345), dpi=80)
    plt.plot(phi, np.log10( MSE_train ), color='green', label="MSE_TRAIN")
    plt.plot(phi, np.log10( MSE_test  ), "--", color='red', label="MSE_TEST")
    if sklcompare == True:
        plt.plot(phi,np.log10(MSE_sklTrain), color="blue", label="SKL_TRAIN")
        plt.plot(phi,np.log10(MSE_sklTest), "--", color="orange", label="SKL_TEST")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"log10(MSE)")
    plt.title(title)
    plt.legend()
    # plt.savefig(f"results/initial/MSE_{title}.pdf", format='pdf', bbox_inches='tight')
    # plt.savefig(f"results/OwnCodeVsSKL/MSE_{title}.pdf", format='pdf', bbox_inches='tight')
    # plt.savefig(f"MSE_{title}.pdf", format='pdf', bbox_inches='tight')
    # plt.savefig(f"results/scaled/MSE_{title}.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # # R2 plot
    # plt.figure(figsize=set_size(345), dpi=80)
    # plt.plot(phi,r2train, color='green', label="R2_TRAIN")
    # plt.plot(phi,r2test, "--", color='red', label="R2_TEST")
    # if sklcompare == True:
    #     plt.plot(phi,R2_sklTrain, color="blue", label="SKL_TRAIN")
    #     plt.plot(phi,R2_sklTest, "--", color="orange", label="SKL_TEST")
    # plt.xlabel(r"$\Phi$")
    # plt.ylabel(r"$R^2$")
    # plt.title(title)
    # plt.legend()
    # # plt.savefig(f"results/initial/R2_{title}.pdf", format='pdf', bbox_inches='tight')
    # # plt.savefig(f"results/OwnCodeVsSKL/R2_{title}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

def beta_plot(noisy):
    plt.style.use("ggplot") 
    data = pd.read_csv(f'results/BetaValsDegNoise:{noisy}.csv')
    plt.figure(figsize=set_size(345), dpi=80)
    marc = ['o','s','^','*','d']
    for i in range(data.shape[0]):
        betdeg = np.array(data.iloc[i])
        betdeg = betdeg[~np.isnan(betdeg)]
        plt.plot(np.arange(0,len(betdeg)),betdeg, marker=marc[i], label=f'max_deg={i}')
    plt.xlabel('Polynomial Variable')
    plt.ylabel(r'Beta Value $\beta$')
    plt.legend()
    # plt.savefig(f"results/betaplot5_noise-{noisy}.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def Figure7(phi,N_b,msetest,msesamp,bias,var,title):
    plt.style.use("ggplot") 
    plt.figure(figsize=set_size(345), dpi=80)
    plt.plot( phi, np.log10( msetest ) , color='blue', label="MSE_test" )
    plt.plot( phi, np.log10( np.mean(msesamp,axis=1,keepdims=True) ) , color='orange', label="MSE_bootsamp" )
    plt.plot( phi, np.log10( bias ) , color='red', label="Bias" )
    plt.plot( phi, np.log10( var ) , color='green', label="Variance" )
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"log10(MSE)")
    plt.title(title+f' N_b:{N_b}')
    plt.legend()
    # plt.savefig(f'results/olsboots.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def Figure8(phi,N_k,cvtest,sklcv,title):
    cvtest = np.squeeze(cvtest)
    plt.style.use("ggplot") 
    plt.figure(figsize=set_size(345), dpi=80)
    plt.plot(phi, np.log10( cvtest  ), "--", color='red', label="cv_MSE")
    plt.plot(phi, np.log10( sklcv  ), color='green', label="cv_scikit")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"log10(MSE)")
    plt.title(title+f' K-Folds:{N_k}')
    plt.legend()
    # plt.savefig(f'results/{N_k}KFOLD_'+title+'.pdf', format='pdf', bbox_inches='tight')
    minind = np.argmin(cvtest)
    print("deg:",minind+1)
    print("mse:",cvtest[minind])
    plt.show()

def Figure9(phi,bmse,cvtest):
    bmse = np.mean( bmse,axis=1,keepdims=True )
    plt.style.use("ggplot") 
    plt.figure(figsize=set_size(345), dpi=80)
    plt.plot(phi, np.log10( bmse  ), label="bt_MSE")
    plt.plot(phi, np.log10( cvtest  ), label="cv_MSE")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"log10(MSE)")
    plt.title(f'100_Boots vs. 20_K-Folds')
    plt.legend()
    # plt.savefig(f'results/bootsvscrossvalid.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def FinalPlot(phi,cvtest,df,Nlams):
    olscv = cvtest
    ridgecv = np.loadtxt(f'{df}/ridge/cvtest_Nlmb{Nlams}.txt')
    lassocv = np.loadtxt(f'{df}/lasso/cvtest_Nlmb{Nlams}.txt')
    if df == 'frank':
        r_opt = 38
        l_opt = 30
    elif df == 'real':
        r_opt = 4
        l_opt = 11
    plt.style.use("ggplot")
    plt.figure(figsize=set_size(345), dpi=80)
    plt.plot(phi, np.log( olscv ), label="OLS")
    plt.plot(phi, np.log( ridgecv[:,r_opt] ), label="Ridge")
    plt.plot(phi, np.log( lassocv[:,l_opt] ), label="Lasso")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"log10(MSE)")
    # plt.title(f"Regression Methods on FrankeFunction")
    plt.title("Regression Methods on Real Terrain")
    plt.legend()
    # plt.savefig(f'results/FrankeAll3.pdf',format='pdf',bbox_inches='tight')
    # plt.savefig(f'results/RealAll3.pdf',format='pdf',bbox_inches='tight')
    if df == 'real':
        print("MSEols:", olscv[3])
        print("MSEridge:", ridgecv[3,r_opt])
        print("MSElasso:", lassocv[2,l_opt])
    elif df == 'frank':
        print("MSEols:", olscv[2])
        print("MSEridge:", ridgecv[5,r_opt])
        print("MSElasso:", lassocv[2,l_opt])
    plt.show()
