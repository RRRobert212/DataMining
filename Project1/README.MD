The file RobertDonohue-HW1-DataMining.pdf contains answers to all questions as given in the assignment.

The code is also included as four python files.

NonCrossValidation.py is code to answer question 1

CrossValidation.py is code to answer question 2

LearningCurve.py is code to answer question 3

LinearRegression.py is a library of functions upon which all 3 of the above files depend on. This is the brunt of the project, the other files are just main() functions.

To run, simply type 'python NonCrossValidation.py', or replace with any of the other 2 filenames, in your terminal. Ensure that you have numpy and matplotlib files installed. Ensure that you have the HW1_dataset folder and CSV files in your current directory and all filenames/paths are the same. Example path is 'HW1_dataset/train-1000-100.csv'. Everything was written and executed in Linux and I know Windows uses \ instead of / but I think Python handles this automatically? if not ctrl+f replace them I guess...

When running NonCrossValidation.py, the plots will appear 1 by 1. Just click the exit button for the next one to appear. There should be 9 in total. 6 for each training set, and 3 for the subset training sets with lambda = 0 omitted. For both NonCrossValidation.py and CrossValidation.py, information about lambda and MSE is printed to the terminal.

Do not run LinearRegression.py, it won't do anything.


