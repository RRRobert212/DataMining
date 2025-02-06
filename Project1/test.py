import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('HW1_dataset/train-1000-100.csv')  # Replace with your actual file name

# All x variables (every column except last)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)  # Target

# 80/20 Train-Test Split
split_index = int(0.8 * len(df))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Function to normalize data
#check this against lecture notes, normalization function <<---
def normalize(X, mean, std, lambd=1e-10):
    return (X - mean) / (std + lambd)

# Linear Regression using Normal Equation with L2 Regularization
#most important function, understand each line
def linear_regression(X, y, lambd):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
    I = np.eye(X_bias.shape[1])
    I[0, 0] = 0  # Do not regularize the bias term
    
    # Use pseudoinverse for numerical stability
    return np.linalg.pinv(X_bias.T @ X_bias + lambd * I) @ X_bias.T @ y

# Evaluation metric: Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true.flatten() - y_pred.flatten()) ** 2)

# Calculate mean and std on the training data
mean_train = np.mean(X_train, axis=0)
std_train = np.std(X_train, axis=0)

# Handle zero standard deviation
std_train[std_train == 0] = 1e-10

# Normalize training and test data using the training mean and std
X_train_norm = normalize(X_train, mean_train, std_train)
X_test_norm = normalize(X_test, mean_train, std_train)

# Store MSEs for different lambda values
lambdas = range(0,200)  # Wider range for lambdas, including very small values
mse_train_values = []
mse_test_values = []

for lambd in lambdas:
    # Ensure lambda is never exactly zero
    theta = linear_regression(X_train_norm, y_train, lambd)
    
    # Add bias term for prediction
    X_train_bias = np.hstack((np.ones((X_train_norm.shape[0], 1)), X_train_norm))
    X_test_bias = np.hstack((np.ones((X_test_norm.shape[0], 1)), X_test_norm))
    
    y_train_pred = X_train_bias @ theta
    y_test_pred = X_test_bias @ theta

    mse_train = mse(y_train, y_train_pred)
    mse_test = mse(y_test, y_test_pred)

    mse_train_values.append(mse_train)
    mse_test_values.append(mse_test)

# Plot MSE vs Lambda
plt.figure(figsize=(10, 6))
plt.plot(lambdas, mse_train_values, marker='o', label='Train MSE')
plt.plot(lambdas, mse_test_values, marker='s', label='Test MSE')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Lambda (Train and Test)')
plt.legend()
plt.grid()
plt.show()