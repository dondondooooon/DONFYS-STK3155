# legs.py
# For all plots and prints
from header import *

# Plot the simple 1D function
def simple1D(x,func): 
    plt.plot(x,func)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("The Function")
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
    plt.savefig(f'results/{N_k}KFOLD_'+title+'.pdf', format='pdf', bbox_inches='tight')
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
    


