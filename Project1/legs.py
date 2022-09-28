# legs.py
# For all plots and prints

from header import *

# Plot the actual function
def function_show(x,func):
    plt.style.use("fivethirtyeight") 
    plt.plot(x,func)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("The Function")
    plt.show() 
    
# TARONGA ANG \n 

# Plot MSE and R2 as function of complexity + print MSE info
def mse_comp(MSE_train,MSE_test,MSE_sklTrain,MSE_sklTest,r2train,\
    r2test,R2_sklTrain,R2_sklTest,phi,printed,sklcompare,title):
    if printed == True: # Print Facts
        print("\nThe complexity with the min. MSE in training:",phi[np.argmin(msetrain)])
        print("The complexity with the min. MSE in test",phi[np.argmin(msetest)], "\n")
        print("MSE_TRAIN: ", MSE_train, "\n")
        print("MSE_TEST: ", MSE_test, "\n")
        print("R2_TRAIN: ", r2train, "\n")
        print("R2_TEST: ", r2test, "\n")
        if sklcompare == True:
            print("Algo. Train Diff.: ", MSE_sklTrain-MSE_train, "\n")
            print("Algo. Test Diff.: ", MSE_sklTest-MSE_test, "\n\n\n")
            print("Algo. Train Diff.: ", R2_sklTrain-r2train, "\n")
            print("Algo. Test Diff.: ", R2_sklTest-r2test, "\n")

    # MSE plot
    plt.style.use("fivethirtyeight") 
    plt.plot(phi,np.log(MSE_train), color='green', label="MSE_TRAIN")
    plt.plot(phi,np.log(MSE_test), "--", color='red', label="MSE_TEST")
    if sklcompare == True:
        plt.plot(phi,np.log(MSE_sklTrain), color="blue", label="SKL_TRAIN")
        plt.plot(phi,np.log(MSE_sklTest), "--", color="orange", label="SKL_TEST")
    plt.xlabel(r"$\Phi$")
    plt.ylabel(r"ln(MSE)")
    # plt.savefig("plots/degree="+str(n)+"N="+str(N)+"simple.pdf")
    plt.title(title)
    plt.legend()
    plt.show()

    # R2 plot
    plt.plot(phi,r2train, color='green', label="R2_TRAIN")
    plt.plot(phi,r2test, "--", color='red', label="R2_TEST")
    if sklcompare == True:
        plt.plot(phi,R2_sklTrain, color="blue", label="SKL_TRAIN")
        plt.plot(phi,R2_sklTest, "--", color="orange", label="SKL_TEST")
    plt.xlabel(r"$\Phi$")
    plt.ylabel(r"R2")
    # plt.savefig("plots/degree="+str(n)+"N="+str(N)+"simple.pdf")
    plt.legend()
    plt.show()