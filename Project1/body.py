from cProfile import label
from header import *    # Import header file

# Initialize Values & Containers
np.random.seed(69420)
#noise = np.random.uniform(0,1,N)
n = 3
N = 500
x = np.sort(np.random.uniform(0,1,N))
#y = np.sort(np.random.uniform(0,1,N))
func = simple_function(x)
MSE = np.zeros(n)
phi = np.arange(1,n+1)

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split\
     (x,func, test_size = 0.2) # Splitting the Data ## degree arg = 5

# Training Algorithm
X = create_X(X_train,0,n,True) # Design Matrix for Train data 
beta = mylinreg(X,y_train) # Beta 
ytilde_train = X @ beta # Model Function

# Testing
Xtesting = create_X(X_test,0,n,True) # Design Matrix for Test data
ytilde_test = Xtesting @ beta

#checker
plt.plot(x,simple_function(x))
plt.plot(X_train, ytilde_train,'x')
plt.plot(X_test,ytilde_test,'x')
plt.show()
print(MSE_func(fx,ytilde(X,fx)))
# print(mylinreg(X,y_train))

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

# # # scale
# # scaler = StandardScaler()
# # scaler.fit(xtrain)
# # xtrainscaled = scaler.transform(xtrain)
# # xtestscaled = scaler.transform(xtest)

# # haha = ytilde(xtrainscaled,ytrain)
# # print(cock.shape)
# # xtrain, xtest, ytrain, ytest = train_test_split(x,fx,test_size=0.2)
# # plt.plot(x,ytilde)
# # plt.show()



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

# Generate a plot showing MSE as a function of complexity
# plt.plot(phi,MSE)
# plt.xlabel(r'$\Phi$')
# plt.ylabel(r'$MSE$')
# plt.show()
# print(MSE)

# Generate a plot comparing the experimental with the fitted values values.
# fig, ax = plt.subplots()
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$f(x)$')
# ax.plot(x, fx, label=r'f(x)_Data')
# ax.plot(x, polyapr[:,4], label=r'f(x)_Fit')
# ax.legend()
# plt.show()
# plt.plot(x,polyapr[:,2])
# plt.show()
