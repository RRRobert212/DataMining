import os
import pandas as pd
from analysis.matrix import constructMatrix

# list all your log files
log_dir = 'PokerLogs'
log_files = [os.path.join(log_dir, filename) for filename in os.listdir(log_dir) if filename.endswith('.csv')]

# to collect everything
all_amount_matrices = []
all_yesorno_matrices = []

for file in log_files:
    matrixAmount, matrixYesOrNo = constructMatrix(file)
    
    all_amount_matrices.append(matrixAmount)
    all_yesorno_matrices.append(matrixYesOrNo)

# now combine them
big_matrix_amount = pd.concat(all_amount_matrices, ignore_index=True)
big_matrix_amount = big_matrix_amount[(big_matrix_amount['Net Profit'] != 0) & (big_matrix_amount['Net Profit'].notna())]
big_matrix_amount = big_matrix_amount[(big_matrix_amount['Total Hands'] > 10) & (big_matrix_amount['Total Hands'].notna())]

big_matrix_yesorno = pd.concat(all_yesorno_matrices, ignore_index=True)
big_matrix_yesorno = big_matrix_yesorno[(big_matrix_yesorno['Profit Yes or No'] != -1) & (big_matrix_yesorno['Profit Yes or No'].notna())]
big_matrix_yesorno = big_matrix_yesorno[(big_matrix_yesorno['Total Hands'] > 10) & (big_matrix_yesorno['Total Hands'].notna())]

# make sure folders exist
os.makedirs('profitAmountCSVs', exist_ok=True)
os.makedirs('profitYesOrNoCSVs', exist_ok=True)

# save the big matrices
big_matrix_amount.to_csv('data/all_data_amount.csv', index=False)
big_matrix_yesorno.to_csv('data/all_data_yesorno.csv', index=False)
