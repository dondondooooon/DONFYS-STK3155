# legs.py
# For all plots and prints

from header import *

# Function for performing Ridge given a lambda
def Ridgelinreg(X,f,lmb):
    I = np.eye(X.shape[1],X.shape[1])
    A = np.linalg.pinv(X.T @ X + lmb*I) # SVD inverse
    beta = A @ X.T @ f
    return beta # Returns optimal beta 

# MSE via Ridge
def Ridge_learning(x,y,n,func,phi,lmbd):
    # Initiate Containers
    MSE_test = np.zeros(n)
    MSE_train = np.zeros(n)
    MSE_sklTest = np.zeros(n)
    MSE_sklTrain = np.zeros(n)

    # Loop from polydeg = 1 to maxpolydeg
    for degree in phi:
        X = create_X(x,y,degree)
        # Splitting the Data
        X_train, X_test, y_train, y_test = train_test_split\
            (X,func, test_size = 0.2)#, random_state=69)
        for l in lmbd:
            print("")
    return 0

"""
-----------------
"""

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

# Plot the simple 1D function
def simple1D(x,func): 
    plt.plot(x,func)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("The Function")
    plt.show() 

# Plot the Franke function
def frankee(x,y,noise,noisy):
    x,y = np.meshgrid(x,y)
    z = FrankeFunction(x,y,noise,noisy)
    fig = plt.figure(figsize=set_size(345), dpi=80)
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z,\
         cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_xlabel('x', linespacing=3.2)
    ax.set_ylabel('y', linespacing=3.1)
    ax.set_zlabel('z', linespacing=3.4)
    # Add colorbar
    fig.colorbar(surf, shrink=0.5)
    # plt.savefig(f"results/FrankFunction_Noise:{noisy}.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    
# Plot MSE and R2 as function of complexity + print MSE info
def mse_comp(MSE_train,MSE_test,MSE_sklTrain,MSE_sklTest,r2train,\
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
    plt.plot(phi, np.log( MSE_train ), color='green', label="MSE_TRAIN")
    plt.plot(phi, np.log( MSE_test  ), "--", color='red', label="MSE_TEST")
    if sklcompare == True:
        plt.plot(phi,np.log(MSE_sklTrain), color="blue", label="SKL_TRAIN")
        plt.plot(phi,np.log(MSE_sklTest), "--", color="orange", label="SKL_TEST")
    plt.xlabel(r"$\Phi$")
    plt.ylabel(r"ln(MSE)")
    plt.title(title)
    plt.legend()
    # plt.savefig(f"results/initial/MSE_{title}.pdf", format='pdf', bbox_inches='tight')
    # plt.savefig(f"results/mse as function of n,N,noise/MSE_{title}.pdf", format='pdf', bbox_inches='tight')
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
    # plt.ylabel(r"R2")
    # plt.title(title)
    # plt.legend()
    # # plt.savefig(f"results/initial/R2_{title}.pdf", format='pdf', bbox_inches='tight')
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
    # plt.savefig(f"results/betaplot5_noise:{noisy}.pdf", format='pdf', bbox_inches='tight')
    plt.show()

# Plot OLS_boostrap results
def bOLSplot(phi,mse,bias,var):
    
    plt.plot(phi, np.log( np.mean(mse, axis=1, keepdims=True) ) , label='MSE')
    plt.xlabel("Degree")
    plt.ylabel("ln(MSE)")
    plt.plot(phi, np.log( np.mean(bias, axis=1, keepdims=True) ) , label='bias')
    plt.plot(phi, np.log( np.mean(var, axis=1, keepdims=True)  ) , label='Variance')
    plt.legend()
    plt.show()

    # for degree in phi:
    #     t = mse[degree-1,:]
    #     # print(np.mean(t))
    #     # 200 = # of bins
    #     # the histogram of the bootstrapped data (normalized data if density = True)
    #     n, binsboot, patches = plt.hist(t, 50, density=True, facecolor='red', alpha=0.75)
    #     # add a 'best fit' line  
    #     y = norm.pdf(binsboot, np.mean(t), np.std(t))
    #     lt = plt.plot(binsboot, y, 'b', linewidth=1)
    #     plt.xlabel('MSE')
    #     plt.ylabel('Frequency')
    #     plt.grid(True)
    #     plt.title(f"Polydeg={degree}")
    #     plt.show()