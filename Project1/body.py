from header import *     # Import header file
from legs import *       # Import 
# commandline_check()
np.random.seed(6969) # Setting Random Seed constant for all the runs

# Read Commandline Arguments 
parser = argparse.ArgumentParser(description="Choose which function for data")
parser.add_argument('-df','--data_func', metavar='', required=True, help='Data Function')
parser.add_argument('-n','--max_degree', metavar='', required=True, help='Max Polynomial Degree')
parser.add_argument('-N','--data_points', metavar='', required=True, help='Number of Datapoints')
parser.add_argument('-eps','--noise_data', metavar='', required=True, help='Add Epsilon Noise?')
parser.add_argument('-sc','--scaled_data', metavar='', required=True, help='Scale the data?')
parser.add_argument('-p','--print_results', metavar='', required=True, help='Print Results?')
parser.add_argument('-skl','--skl_compare', metavar='', required=True, help='Compare to SciKitLearn')
args = parser.parse_args()
# Initialize Values & Containers
### can prolly turn arguments to bool and save 4 lines of code
n = int(args.max_degree) # Max Polynomial Degree
N = int(args.data_points) # Number of datapoints
noisy = args.noise_data.lower() == "noisy"
scaling = args.scaled_data.lower() == "scale"
printed = args.print_results.lower() == "print"
sklcompare = args.skl_compare.lower() == "compare"
title = f"n:{n}; N:{N}; Noise:{noisy}"
phi = np.arange(1,n+1)
noise = np.random.randn(N) # np.random.uniform(0,1,N) # Random Noise
x = np.sort(np.random.uniform(0,1,N)) 
y = np.sort(np.random.uniform(0,1,N))
# Data_Function
if args.data_func.lower() == "simple": # Easy 1D Function
     func = simple_function(x,noise,noisy) 
elif args.data_func.lower() == "frank": # Frank Function
     func = FrankeFunction(x,y)
else:
     print('Command Line Error: Check your command line arguments\n',\
          '\n Especially for -df DATA_FUNC')
     exit(1)

msetrain, msetest, mskltrain, mskltest, r2train, r2test,\
     rskltrain, rskltest = complexity_dependencies(x,y,n,func,phi,scaling) # OLS MSE [from header]
function_show(x,func)    # Show the actual function [from legs]
mse_comp(msetrain,msetest,mskltrain,mskltest,r2train,r2test,\
     rskltrain,rskltest,phi,printed,sklcompare,title) # Plots + Prints [from legs]

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