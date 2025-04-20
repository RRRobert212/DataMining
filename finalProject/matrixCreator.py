#NOTE THIS PROGRAM CREATES TWO MATRICES
#one has the profit amount in dollars as the target variable, I originally tried guessing the profit, that was impossible
#but I kept it because it's still an interesting matrix
#the matrix we are interested in for the classification model is big_matrix_yesorno, called all_data_yesorno.csv

import os
import pandas as pd
from analysis.matrix import constructMatrix

#access folder containig all the log files
log_dir = 'PokerLogs'
log_files = [os.path.join(log_dir, filename) for filename in os.listdir(log_dir) if filename.endswith('.csv')]


all_amount_matrices = []
all_yesorno_matrices = []

for file in log_files:
    matrixAmount, matrixYesOrNo = constructMatrix(file)
    
    all_amount_matrices.append(matrixAmount)
    all_yesorno_matrices.append(matrixYesOrNo)

#combine them
big_matrix_amount = pd.concat(all_amount_matrices, ignore_index=True)
big_matrix_amount = big_matrix_amount[(big_matrix_amount['Net Profit'] != 0) & (big_matrix_amount['Net Profit'].notna())]
big_matrix_amount = big_matrix_amount[(big_matrix_amount['Total Hands'] > 10) & (big_matrix_amount['Total Hands'].notna())]

big_matrix_yesorno = pd.concat(all_yesorno_matrices, ignore_index=True)
big_matrix_yesorno = big_matrix_yesorno[(big_matrix_yesorno['Profit Yes or No'] != -1) & (big_matrix_yesorno['Profit Yes or No'].notna())]
big_matrix_yesorno = big_matrix_yesorno[(big_matrix_yesorno['Total Hands'] > 10) & (big_matrix_yesorno['Total Hands'].notna())]
big_matrix_yesorno = big_matrix_yesorno[(big_matrix_yesorno['Total Calls'] > 1) & (big_matrix_yesorno['Total Calls'].notna())]

#you need a 'data' folder
os.makedirs('data', exist_ok=True)

# save them to data folder
big_matrix_amount.to_csv('data/all_data_amount.csv', index=False)
big_matrix_yesorno.to_csv('data/all_data_yesorno.csv', index=False)
