from header import *     # Import header file
from ridge import *
from legs import *       # Import 

# commandline_check()

np.random.seed(2018) # Setting Random Seed constant

# Read Commandline Arguments 
parser = argparse.ArgumentParser(description="Choose which function for data")
parser.add_argument('-df','--data_func', metavar='', required=True, help='Data Function')
parser.add_argument('-n','--max_degree', metavar='', required=True, help='Max Polynomial Degree')
parser.add_argument('-N','--data_points', metavar='', required=True, help='Number of Datapoints')
parser.add_argument('-eps','--noise_data', metavar='', required=True, help='Add Epsilon Noise?')
parser.add_argument('-p','--print_results', metavar='', required=True, help='Print Results?')
parser.add_argument('-skl','--skl_compare', metavar='', required=True, help='Compare to SciKitLearn')
args = parser.parse_args()
# Initialize Values & Containers
### can prolly turn arguments to bool and save 4 lines of code
n = int(args.max_degree) # Max Polynomial Degree
N = int(args.data_points) # Number of datapoints
noisy = args.noise_data.lower() == "noisy"
printed = args.print_results.lower() == "print"
sklcompare = args.skl_compare.lower() == "compare"
title = f"n:{n}; N:{N}; noise:{noisy}"
phi = np.arange(1,n+1)
noise = np.random.uniform(0,1,N) # Random Noise
x = np.sort(np.random.uniform(0,1,N)) 
y = np.sort(np.random.uniform(0,1,N))
# Data_Function
if args.data_func.lower() == "simple": # Easy 1D Function
     func = simple_function(x,noise,noisy) 
     y = 0
elif args.data_func.lower() == "frank": # Frank Function
     func = FrankeFunction(x,y,noise,noisy)
else:
     print('Command Line Error: Check your command line arguments\n',\
          '\n Especially for -df DATA_FUNC')
     exit(1)
N_bs = 100 # make argpars later
nlambdas = 5 #10 # 100
lambdas = np.logspace(-4,4,nlambdas)

# bmse, bbias, bvar = OLS_boots(x,y,n,func,phi,N_bs)

# # Main magic happenings 
msetrain, msetest, mskltrain, mskltest, r2train, r2test,\
     rskltrain, rskltest = OLS_learning(x,y,n,func,phi,noisy) # OLS MSE [from header]
bmse, bbias, bvar = OLS_boots(x,y,n,func,phi,N_bs)
# ridge_msetrain, ridge_msetest, ridgeskltrain,\
#      ridgeskltest = Ridge_learning(x,y,n,func,phi,nlambdas,lambdas)
# rmsesamp = Ridge_Boots(x,y,n,func,phi,nlambdas,lambdas,N_bs)
# plt.imshow(np.mean(rmsesamp,axis=0,keepdims=True).reshape(n,nlambdas))
# plt.colorbar()
# plt.show()
# msetest = np.mean(rmsesamp,axis=0,keepdims=True).reshape(n,nlambdas)
# min_ind = np.argmin(msetest)
# mrow, mcol = np.where(msetest == msetest.ravel()[min_ind])
# print("The lowest MSE is found at: Row:",mrow, "and Col:", mcol)
# print(phi[mrow])
# print(lambdas[mcol])



# # Plot Functions
# simple1D(x,func)  # Plot 1D simple function
# frankee(x,y,noise,noisy) # Plot 2D Franke Function
# ols_first(msetrain,msetest,mskltrain,mskltest,r2train,r2test,\
#      rskltrain,rskltest,phi,printed,sklcompare,title) # Plots + Prints OLS no sampling
# beta_plot(noisy) # Plot beta evolution in complexity
bOLSplot(phi,bmse,bbias,bvar,msetest,N_bs)
# RidgePlot(ridge_msetrain,ridge_msetest,ridgeskltrain,\
#      ridgeskltest,phi,lambdas,title)

# plt.plot(np.log(np.mean(bmse,axis=1,keepdims=True)))
# plt.show()