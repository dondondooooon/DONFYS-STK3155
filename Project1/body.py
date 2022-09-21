from header import *    # Import header file
commandline_check()

# Initialize Values & Containers
np.random.seed(69420) # Setting Random Seed constant for all the runs
n = int(sys.argv[1]) # Max Polynomial Degree
N = int(sys.argv[2]) # Number of datapoints
MSE_train = MSE_test = r2train = r2test = np.zeros(n)
phi = np.arange(1,n+1)
noise = np.random.uniform(0,1,N) # Random Noise
x = np.sort(np.random.uniform(0,1,N)) 
y = np.sort(np.random.uniform(0,1,N))
func = simple_function(x) #* noise # 1D tryout
# func = FrankeFunction # 2D Frank Function

# Plotting MSE and R2 as functions of Complexity 
for degree in phi: # skipped 0th complexity 
     print(degree)
     X = create_X(x,0,degree,True) # Build Design Matrix
     # Splitting the Data
     X_train, X_test, y_train, y_test = train_test_split\
          (X,func, test_size = 0.2, random_state=69) 

# SCALE HERE B4 FITTING :: so it'd be mylinreg(X_train_scaled, y_train)

     # Training 
     beta = mylinreg(X_train,y_train) # Beta 
     ytilde_train = X_train @ beta # Model Function
     # Testing
     ytilde_test = X_test @ beta
     # MSE 
     MSE_train[degree-1] = MSE_func(y_train,ytilde_train)
     MSE_test[degree-1] = MSE_func(y_test,ytilde_test)
     r2train[degree-1] = R2(y_train,ytilde_train)
     r2test[degree-1] = R2(y_test,ytilde_test)
     # print("MSE_TRAIN: ", MSE_train[degree-1])
     # print("MSE_TEST: ", MSE_test[degree-1])
     # print("Diff: ", MSE_train[degree-1]-MSE_test[degree-1])

print("MSE_TRAIN: ", MSE_train)
print("MSE_TEST: ", MSE_test)
print("Diff: ", MSE_train-MSE_test)
plt.plot(phi,np.log(MSE_train), label="MSE_TRAIN")
plt.plot(phi,np.log(MSE_test),"x", label="MSE_TEST")
plt.xlabel(r"$\Phi$")
plt.ylabel(r"ln(MSE)")
# plt.savefig("plots/degree="+str(n)+"N="+str(N)+"simple.pdf")
plt.legend()
plt.show()

print("R2_TRAIN: ", r2train)
print("R2_TEST: ", r2test)
print("Diff: ", r2train-r2test)
plt.plot(phi,r2train, label="R2_TRAIN")
plt.plot(phi,r2train, label="R2_TEST")
plt.xlabel(r"$\Phi$")
plt.ylabel(r"R2")
# plt.savefig("plots/degree="+str(n)+"N="+str(N)+"simple.pdf")
plt.legend()
plt.show()


#checker
# print("MSE TRAIN:", MSE_func(y_train,ytilde_train))
# print("MSE TEST:", MSE_func(y_test,ytilde_test))
# plt.plot(x,simple_function(x))
# plt.plot(X_train, ytilde_train,'x')
# plt.plot(X_test,ytilde_test)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()



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



# PLOTTING COMPLEXITY(DEGREE) 
# for degree in range(1,n+1): # skipped 0th complexity kekw
#     X = create_X(X_train,0,degree,True)
#     fx = y_train
#     y = ytilde(X,fx)
#     MSE[degree-1] = MSE_func(fx,y)
# # print(MSE)
# plt.plot(phi,np.log(MSE))
# plt.xlabel(r"ln$\Phi$")
# plt.ylabel(r"MSE")
# plt.show()
# print(MSE)