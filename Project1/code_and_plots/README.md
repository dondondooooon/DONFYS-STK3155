header.py
--------
This is python script is where most of the imported libraries are called for the regression analysis, and also where most of the functions used for them are defined. Among these functions are: generating Franke function, processing terrain data, generating the design matrix, finding optimal beta, calculating MSE and R^2 score, scaling the data. The functions for plotting some of the results are also in this file.

--------------------------------------------------

main.py
--------
A python script that takes in command line arguments for certain parameters in the regression analysis. One can run "python3 -h" for help with these argparses. These shorthand commandline arguments are as follows:

Required arguments:
  -df  "frank" to analyze the Franke function or "real" to analyze terrain data
  -n   Max degree polynomial you wish model
  -N   Max of data points
  -scl "noscale" to not scale the data or "yescale" to scale the data
  -reg "ols" for OLS regression, "ridge" for Ridge regression, "lasso" for Lasso regression
  -nb  Number of bootstrap iterations
  -nk  Number of K-folds in cross-validation
Optional arguments:
  -q  Specify to have no noise in data set (just for Franke function)
  -pr Specify to print the plot of the chosen data function
  -optlam  Specify to have an array of lambda

Run command:
Example of build command for regression analysis on Franke function and print how the function looks like:
  >> python3 main.py -df frank -n 10 -N 50 -scl noscale -reg ols -nb 100 -nk 20 -pr
Example of build command for regression analysis on Topographic data set:
  >> python3 main.py -df real -n 10 -N 80 -scl yescale -reg ols -nb 100 -nk 20


To print out the respective plots in the figures used in the report, simply uncomment the corresponding function in the lower part of the script (lines 201 to 208). The plots are usually saved (if the plt.savefig line is uncommented) in the folder "results"

Note!: -optlam is used in assumption that the regression being used is ridge and lasso and in conjunction that the last part of the script (lines 211 to 216) is uncommented such that the big matrices of MSE scores with varying complexity and lambda values used is exported as .txt files into the matrices directory.

--------------------------------------------------

matrices
--------
This folder holds the .txt files for the exported files of the matrices containing the MSE scores of both the Franke function and terrain data as one varies the complexity of the polynomial degree of the model and lambda parameter used.

--------------------------------------------------

lmb_read.py
--------
This python script reads the .txt files in the matrices folder and prints the necessary heat maps and plots for Ridge and Lasso Regression. This method was done to avoid having to run the entire program for analyzing the same data set repeatedly.

Run command: python3 lmb_read.py

results
--------
This folder holds all the plots and figures generated during our investigation.