# legs.py
# For all plots and prints

from header import *
from ridge import *
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
    # plt.savefig(f"results/betaplot5_noise:{noisy}.pdf", format='pdf', bbox_inches='tight')
    plt.show()

# Plot OLS_boostrap results
def bOLSplot(phi,mse,bias,var,mseols,nboot):
    plt.plot(phi, np.log10( mseols), label='MSE_test')
    plt.plot(phi, np.log10(  np.nanmean(mse, axis=1, keepdims=True) ) , label='MSE_samp')
    plt.xlabel(r"\Phi")
    plt.ylabel("log10(MSE)")
    plt.plot(phi, np.log10( bias)  , label='Bias')
    plt.plot(phi, np.log10( var ), label='Variance')
    plt.legend()
    plt.title(f"n=10; N=50; N_b:{nboot}")
    plt.savefig(f"results/olsboots.pdf", format='pdf', bbox_inches='tight')
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

# Plot Ridge or Lasso plot
def RidgePlot(msetrain,msetest,sklmsetrain,sklmsetest,phi,lambdas,title):
    # Find Index of Minimum MSE as function of polynomial and lambda
    min_ind = np.argmin(msetest)
    mrow, mcol = np.where(msetest == msetest.ravel()[min_ind])
    print("The lowest MSE is found at: Row:",mrow, "and Col:", mcol)

    # # Plot colormap of MSE as function of n polydegree and lambda
    # plt.imshow(msetest, cmap=cm.coolwarm)
    # plt.xlabel(r"$\lambda$")
    # plt.ylabel(r"n_degree")
    # cbar = plt.colorbar()
    # cbar.set_label('MSE')
    # plt.title("Test:" + title + f"; Nlambda:{len(lambdas)}")
    # # plt.savefig('results/Ridgestuff/colormapofmsetestridge.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

    # # Plot MSE_Ridge or _Lasso
    # plt.style.use("ggplot")
    # plt.figure(figsize=set_size(345), dpi=80) 
    # for degree in phi:
    #     # plt.plot(np.log10(lambdas), np.log( msetrain[degree-1,:]    ) , color='green', label=f'MSE_train')#: Max_Deg={degree}')
    #     # plt.plot(np.log10(lambdas), np.log( sklmsetrain[degree-1,:] ) , color='blue', label=f'SKL_train')#: Max_Deg={degree}')
    #     plt.plot(np.log10(lambdas), np.log( msetest[degree-1,:]     ) , label= f'Max_Deg={degree}')   # , '--', color='red', label=f'MSE_test')
    #     # plt.plot(np.log10(lambdas), np.log( sklmsetest[degree-1,:]  ) , '--', color='orange', label=f'SKL_test')#: Max_Deg={degree}')
    # plt.xlabel(r"$\lambda$")
    # plt.ylabel(r"ln(MSE)")
    # plt.title('MSE_test; ' + title)
    #     # plt.title(f'Deg_now={degree}; '+ title)
    # plt.legend()
    #     # if degree == 5:
    #     #     plt.savefig('results/OwnCodeVsSKL/RidgeVsSKLn=5.pdf', format='pdf', bbox_inches='tight')
    # # plt.savefig('results/Ridgestuff/mseVSlambdaForDegto10.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

# def BootstrapRidgePlot()
