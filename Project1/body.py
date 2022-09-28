from header import *     # Import header file
from legs import *       # Import 
commandline_check()

# Initialize Values & Containers
np.random.seed(6969) # Setting Random Seed constant for all the runs
n = int(sys.argv[1]) # Max Polynomial Degree
N = int(sys.argv[2]) # Number of datapoints
noisy = sys.argv[3].lower() == "noisy"
scaling = sys.argv[4].lower() == "scale"
printed = sys.argv[5].lower() == "print"
sklcompare = sys.argv[6].lower() == "compare"
title = f"n:{n}; N:{N}; Noise:{noisy}"

phi = np.arange(1,n+1)
noise = np.random.randn(N) # np.random.uniform(0,1,N) # Random Noise
x = np.sort(np.random.uniform(0,1,N)) 
y = np.sort(np.random.uniform(0,1,N))
func = simple_function(x,noise,noisy) #* noise # 1D tryout
# func = FrankeFunction # 2D Frank Function

msetrain, msetest, mskltrain, mskltest, r2train, r2test,\
     rskltrain, rskltest = complexity_dependencies(x,y,n,func,phi,scaling) # [from header]

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