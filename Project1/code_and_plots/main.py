from header import *     # Import header file

np.random.seed(2018) # Setting Random Seed constant

# Read Commandline Arguments 
parser = argparse.ArgumentParser(description="Choose parameters for the program")
parser.add_argument('-df','--data_func', metavar='', required=True, help='Data Function')
parser.add_argument('-n','--max_degree', metavar='', required=True, help='Max Polynomial Degree')
parser.add_argument('-N','--data_points', metavar='', required=True, help='Number of Datapoints')
parser.add_argument('-q','--quiet_data',help='Specify to remove noise',action='store_true')
parser.add_argument('-scl','--scale',metavar='', required=True, help='Scaling?')
parser.add_argument('-reg','--reg_method', metavar='', required=True, help='Regression Method')
parser.add_argument('-nb','--bootsiter', metavar='', required=True, help='Bootstrap Iteration')
parser.add_argument('-nk','--kfold', metavar='', required=True, help='No. of k-Folds')
parser.add_argument('-pr','--print_plot',help='Specify to print',action='store_true')
parser.add_argument('-optlam','--opt_lambda',help='Specify to execute lambda array',action='store_true')
args = parser.parse_args()

# Dictionaries
DataFunc = {"frank": FrankeFunction, 'real': RealFunction} # Output Data Function to be used
RegMethod = {"ols": OLSlinreg, 'ridge': Ridgelinreg, 'lasso': Lassolinreg} # Regression Method to be used
Scaling = {"noscale": noscale, "yescale": scale_data} # To scale or not scale data 

# Initialize Values & Containers
n = int(args.max_degree) # Max Polynomial Degree
N = int(args.data_points) # Number of datapoints
phi = np.arange(1,n+1) # Array of Polynomial Degree
if args.quiet_data == False:
     noise = np.random.uniform(0,1,N) * 0.2 # Random Noise
else:
     noise = 0

# Initialize x,y, and Data Function
x,y,func = DataFunc[args.data_func](N,noise,args.print_plot)

# Resampling params
N_boots = int(args.bootsiter)
N_k = int(args.kfold)
# Lambda params
if args.opt_lambda == True:
     if args.data_func == 'real':
          Nlams = 20 # for terrain
          lambdas = np.logspace(-10,1,Nlams)
     else:
          Nlams = 50 # for franke
          lambdas = np.logspace(-18,1,Nlams)
else:
     lambdas = np.array([0.0001,0.001,0.01,0.1,100])
title = f"n:{n}; N:{N}"

'''
The following lines of codes are the main regression code
with each of their respective resampling method 
'''

def CrossValidResampling(x,y,n,func,phi,lmb,nK):
     msesamp = np.zeros((n,nK))
     tr_msesamp = np.zeros((n,nK))
     sklmse = np.zeros(n)
     ridge = Ridge(alpha = lmb)

     for degree in phi:
          X = create_X(x,y,degree) # Build Design Matrix
          X_test_notused, y_test_notused = np.copy(X), np.copy(func)  # Dummy containers
          X, X_test_notused, func, y_test_notused =\
               Scaling[args.scale](X,X_test_notused,func,y_test_notused) # Scale the Data

          _kfold = KFold(n_splits = nK) # Create folds
          j = 0
          for itr,jte in _kfold.split(X):
               Xtr, ytr = X[itr], func[itr]
               Xte, yte = X[jte], func[jte]
               
               beta = RegMethod[args.reg_method](Xtr,ytr,lmb)

               ytilde = Xtr @ beta
               ypred = Xte @ beta
               tr_msesamp[degree-1,j] = MSE_func(ytr,ytilde)
               msesamp[degree-1,j] = MSE_func(yte,ypred)

               j += 1
          # SciKitLearn
          estimated_mse_folds = cross_val_score(ridge, X, func,\
               scoring='neg_mean_squared_error', cv=_kfold)
          sklmse[degree-1] = np.mean(-estimated_mse_folds)
     # Take mean of all mse in k-fold iterations
     msetrain = np.squeeze( np.mean(tr_msesamp, axis=1, keepdims=True) )
     msetest = np.squeeze( np.mean(msesamp, axis=1, keepdims=True) )
     return msetest, msetrain, sklmse

def BootstrapResampling(x,y,n,func,phi,lmb,nB):
     var = np.zeros(n)
     bias = np.zeros(n)
     msesamp = np.zeros((n,nB))

     for degree in phi:
          X = create_X(x,y,degree) # Build Design Matrix

          X_train, X_test, y_train, y_test = train_test_split\
               (X,func, test_size = 0.2, random_state=1) # Splitting the Data
          X_train, X_test, y_train, y_test =\
               Scaling[args.scale](X_train,X_test,y_train,y_test) # Scale the Data

          ypred = np.empty((y_test.shape[0],nB)) # Initialize predict model array

          for boots in range(0,nB): # Bootstrap Iterations
               X_trboot, y_trboot = bootstrap(X_train,y_train) # Sample Data

               beta = RegMethod[args.reg_method](X_trboot,y_trboot,lmb) # Calculate Beta

               ypred[:,boots] = ypr = (X_test @ beta).ravel()  # Testing
               msesamp[degree-1,boots] = MSE_func(y_test,ypr) # Calculate MSE
          
          # Calculate Bias & Variance
          bias[degree-1] = np.nanmean( (y_test - np.nanmean(ypred, axis=1, keepdims=True))**2 )
          var[degree-1] = np.mean( np.nanvar(ypred, axis=1) )
     return msesamp,bias,var

def NoResampling(x,y,n,func,phi,lmb):
    # Initiate Containers
    degree_beta = []
    r2test = np.zeros(n)
    r2train = np.zeros(n)
    MSE_test = np.zeros(n)
    MSE_train = np.zeros(n)
    R2_sklTest = np.zeros(n)
    R2_sklTrain = np.zeros(n)
    MSE_sklTest = np.zeros(n)
    MSE_sklTrain = np.zeros(n)

    for degree in phi:
        X = create_X(x,y,degree) # Create Design Matrix
        # Split the Data
        X_train, X_test, y_train, y_test = train_test_split\
            (X,func, test_size = 0.2, random_state=1) 
        # Scale the Data
        X_train, X_test, y_train, y_test = Scaling[args.scale](X_train,X_test,y_train,y_test)

        beta = RegMethod[args.reg_method](X_train,y_train,lmb)
        # plot_beta_save(degree_beta,beta) # For plotting beta as increasing order of polynomials 
        
        ytilde = X_train @ beta # Training
        ypredict = X_test @ beta # Testing

        # MSE & R2 score via own Algorithm
        MSE_train[degree-1] = MSE_func(y_train,ytilde)  
        MSE_test[degree-1] =  MSE_func(y_test,ypredict) 
        r2train[degree-1] = R2(y_train,ytilde)
        r2test[degree-1] = R2(y_test,ypredict)

        # SciKitLearnRegCheck [Only OLS check though]
        clf = skl.LinearRegression().fit(X_train,y_train) # fit_intercept=False ?
        MSE_sklTrain[degree-1] = mean_squared_error(clf.predict(X_train),y_train)
        MSE_sklTest[degree-1] = mean_squared_error(clf.predict(X_test),y_test)
        R2_sklTrain[degree-1] = clf.score(X_train,y_train)
        R2_sklTest[degree-1] = clf.score(X_test,y_test)
        # For plotting beta as increasing order of polynomials    
        # BetaValues = pd.DataFrame(degree_beta)
        # BetaValues.to_csv(f'results/BetaValsDegNoise:{noisy}.csv', index=False)
    return (MSE_train,MSE_test,MSE_sklTrain,MSE_sklTest,\
        r2train,r2test,R2_sklTrain,R2_sklTrain)

# Resampling Dictionary 
Resampling = {"noresamp": NoResampling, "bootstrap": BootstrapResampling,\
               "crossvalid": CrossValidResampling} # Resampling Method to be used

# Function for calculating lambda dependency
def lambda_mse(lambdas):
     # Initialize MSE containers
     msetrain, msetest = np.zeros((n,Nlams)), np.zeros((n,Nlams))
     mskltrain, mskltest, r2train, r2test, rskltrain, rskltest\
          = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n) 
     bbias, bbvar = np.zeros((Nlams,n)), np.zeros((Nlams,n))
     bmse = np.zeros((Nlams,n,N_boots))
     cvtest = np.copy(msetest) # np.zeros((Nlams,n,N_k))
     cvtrain = np.copy(msetest) 
     sklmse = np.copy(msetest)
     i = 0
     for lmb in lambdas:
          msetrain[:,i], msetest[:,i], _mskltrain, _mskltest, _r2train, _r2test,\
               _rskltrain, _rskltest = Resampling['noresamp'](x,y,n,func,phi,lmb)
          bmse[i,:,:], bbias[i,:], bbvar[i,:] = Resampling['bootstrap'](x,y,n,func,phi,lmb,N_boots)
          cvtest[:,i], cvtrain[:,i], sklmse[:,i] = Resampling['crossvalid'](x,y,n,func,phi,lmb,N_k)
          i += 1
     _bmse = np.squeeze(np.mean(bmse,axis=2,keepdims=True))
     return msetrain,msetest,_bmse,bbias,bbvar,cvtest,cvtrain,sklmse

# For OLS (i.e. regularization param is 0)
if args.reg_method.lower() == 'ols':
     lmb = 0
     msetrain, msetest, mskltrain, mskltest, r2train, r2test,\
          rskltrain, rskltest = Resampling['noresamp'](x,y,n,func,phi,lmb)
     bmse, bbias, bbvar = Resampling['bootstrap'](x,y,n,func,phi,lmb,N_boots)
     cvtest,cvtrain,sklcv = Resampling['crossvalid'](x,y,n,func,phi,lmb,N_k)
else: # Call lambda dependency
     msetrain, msetest, bmse, bbias, bbvar, cvtest, cvtrain, sklcv =\
          lambda_mse(lambdas)

# Plotting:
# Initial mse plots:
# ols_first(msetrain,msetest,mskltrain,mskltest,r2train,r2test,rskltrain,\
#    rskltest,phi,False,False,title+" noise:True")
# Beta plot:
#beta_plot(False)
# Figure7(phi,N_boots,msetest,bmse,bbias,bbvar,title)
# Figure8(phi,N_k,cvtest,sklcv,title) # Printout my cv vs scitkit learn cv
# Figure9(phi,bmse,cvtest)
# FinalPlot(phi,cvtest,args.data_func,Nlams) # run only on -reg ols 

"""
# For Finding optimal lambda in ridge and lasso
np.savetxt(f'matrices/{args.data_func}/{args.reg_method}/msetrain_Nlmb{Nlams}.txt', msetrain)
np.savetxt(f'matrices/{args.data_func}/{args.reg_method}/msetest_Nlmb{Nlams}.txt', msetest)
np.savetxt(f'matrices/{args.data_func}/{args.reg_method}/bmse_Nlmb{Nlams}.txt', bmse)
np.savetxt(f'matrices/{args.data_func}/{args.reg_method}/cvtest_Nlmb{Nlams}.txt', cvtest)
np.savetxt(f'matrices/{args.data_func}/{args.reg_method}/cvtrain_Nlmb{Nlams}.txt', cvtrain)
"""
