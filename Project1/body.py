from header import *    # Import header file

# Initialize Values & Containers
np.random.seed(69420)
#noise = np.random.uniform(0,1,N)
n = 2
N = 500
x = np.random.rand(100) #np.sort(np.random.uniform(0,1,N))
#y = np.sort(np.random.uniform(0,1,N))
fx = simple_function(x)
MSE = np.zeros(n)
phi = np.arange(1,n+1)

#print(create_X(x,0,n,True))
print(ytilde(create_X(x,0,n,True),fx))

print("x")

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

# X_train, X_test, y_train, y_test = train_test_split\
#     (create_X(x,0,n,True),fx, test_size = 0.2) # Splitting the Data ## degree arg = 5

# MSE = MSE_func(y_train,ytilde(X_train,y_train))



# for degree in range(1,n+1): # skipped 0th complexity kekw
#     polyapr, X_train, X_test, Y_train, Y_test = ytilde(degree,x,fx)
#     MSE = MSE_func(fx)

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


# reg = LinearRegression().fit(X_train,fx_train)
# reg.score(X,fx)