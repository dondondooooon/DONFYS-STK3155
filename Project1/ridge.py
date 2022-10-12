from header import *

# Function for performing Ridge given a lambda
def Ridgelinreg(X,f,lmb):
    I = np.eye(X.shape[1])#,X.shape[1])
    A = np.linalg.pinv(X.T @ X + lmb*I) # SVD inverse
    beta = A @ X.T @ f
    return beta # Returns optimal beta 

# MSE via Ridge
def Ridge_learning(x,y,n,func,phi,nl,lambds):
    # Initiate Containers
    msetest = np.zeros((n,nl))
    msetrain = np.zeros((n,nl))
    mseskltest = np.zeros((n,nl))
    mseskltrain = np.zeros((n,nl))

    # Loop from polydeg = 1 to maxpolydeg
    for degree in phi:
        X = create_X(x,y,degree)
        # Splitting the Data
        X_train, X_test, y_train, y_test = train_test_split\
            (X,func, test_size = 0.2, random_state=1)
        for i in range(nl):
            lam = lambds[i]

            # Own Ridge
            beta = Ridgelinreg(X_train,y_train,lam)
            # ytilde = X_train @ beta
            ypredict = X_test @ beta
            # msetrain[degree-1,i] = MSE_func(y_train,ytilde)
            msetest[degree-1,i] = MSE_func(y_test,ypredict)

            # #ScikitLearn
            # RegRidge = skl.Ridge(lam)
            # RegRidge.fit(X_train,y_train)
            # skl_ytilde = RegRidge.predict(X_train)
            # skl_ypredict = RegRidge.predict(X_test)
            # mseskltrain[degree-1,i] = MSE_func(y_train,skl_ytilde)
            # mseskltest[degree-1,i] = MSE_func(y_test,skl_ypredict)
    return msetrain, msetest, mseskltrain, mseskltest

# Bootstrap Ridge
def Ridge_Boots(x,y,n,func,phi,nl,lambds,nB):
    # Initiate Containers
    var = np.zeros((nB,n,nl))
    bias = np.zeros((nB,n,nl))
    msesamp = np.zeros((nB,n,nl)) # mse test

    for degree in phi:
        X = create_X(x,y,degree)
        X_train, X_test, y_train, y_test = train_test_split\
            (X,func,test_size=0.2,random_state=1)
        ypred = np.empty((y_test.shape[0],nB))
        for i in range(nl):
            lam = lambds[i]
            for boots in range(0,nB):
                Xtrboot, ytrboot = bootstraping(X_train,y_train)
                beta = Ridgelinreg(Xtrboot,ytrboot,lam)
                ypred[:,boots] = ypr = (X_test @ beta).ravel()
                msesamp[boots,degree-1,i] = MSE_func(y_test,ypr)
    return msesamp