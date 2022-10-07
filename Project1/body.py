from header import *     # Import header file
from legs import *       # Import 

# commandline_check()

# np.random.seed(6969) # Setting Random Seed constant for all the runs
np.random.seed(7132)

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

N_bs = 90 # make argpars later

bmse, bbias, bvar = OLS_boots(x,y,n,func,phi,N_bs)

# Main magic happenings
msetrain, msetest, mskltrain, mskltest, r2train, r2test,\
     rskltrain, rskltest = OLS_learning(x,y,n,func,phi,noisy) # OLS MSE [from header]
     
# bmse, bbias, bvar = OLS_boots(x,y,n,func,phi,N_bs)

print(np.mean(bmse, axis=1, keepdims=True))
print("\n",msetest)
da = pd.DataFrame(bmse)
display(da)

# plt.plot(np.log(np.mean(bmse, axis=1, keepdims=True)).ravel())
# plt.xlabel("degree")
# plt.ylabel("ln(MSE)")
# plt.show()

# Plot Functions
# simple1D(x,func)    # Plot 1D simple function [from legs]
# frankee(x,y,noise,noisy)
mse_comp(msetrain,msetest,mskltrain,mskltest,r2train,r2test,\
    rskltrain,rskltest,phi,printed,sklcompare,title) # Plots + Prints [from legs]
# beta_plot(noisy)
# print(msetest,"\n\n")
# print(mskltest,"\n\n")
# print(np.mean(bmse,axis=1,keepdims=True))
bOLSplot(phi,bmse,bbias,bvar)

# import inspect
# def count_positional_args_required(func):
#     signature = inspect.signature(func)
#     empty = inspect.Parameter.empty
#     total = 0
#     for param in signature.parameters.values():
#         if param.default is empty:
#             total += 1
#     return total
# print( count_positional_args_required(ytilde) )