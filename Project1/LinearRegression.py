import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data

def createMatrices(data):
    """X is all sample data, every column but last, y vector is final column"""
    X = data[:, :-1]
    y = data[:, -1]

    #add a column of 1s to X (from lecture slides, each x includes x0 = 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X, y

def linear_regression(X, y, lambda_):
    """main function for regularized linear regression"""

    #numpy function for I matrix
    I = np.eye(X.shape[1])
    #bias term stays 0
    I[0, 0] = 0 

    #equation derived in class, '@' means matrix multiplication and .T means transpose. numpy has built in inverse (.inv)
    w = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
    return w

def compute_mse(X, y, w):
    """based on w obtained from linear_regression, compute MSE"""
    return np.mean((y - X @ w) ** 2)

def cross_validate(X, y, lambdas, k=10):
    """cross validation technique. given k, split data into k folds, each one serves as validation data and mse is found based off it. 
    Avg MSE is computed and best lambda is chosen from best of avg MSEs."""
    fold_size = X.shape[0] // k
    best_lambda = None
    #best_mse is large to start
    best_mse = float('inf')

    #for each lambda, compute mse for each k
    for lambda_ in lambdas:
        mse_values = []
        for i in range(k):
            # Split data into k folds
            k_start = i * fold_size
            if i!= k-1:
                k_end = (i + 1) * fold_size 
            else: k_end = X.shape[0]

            #training data is all data not in fold, so concatenenate it together
            X_train = np.concatenate([X[:k_start], X[k_end:]], axis=0)
            y_train = np.concatenate([y[:k_start], y[k_end:]], axis=0)

            X_val = X[k_start:k_end]
            y_val = y[k_start:k_end]

            #run model on new X_train and y_train
            w = linear_regression(X_train, y_train, lambda_)

            #get mse for this k and append it to list
            val_mse = compute_mse(X_val, y_val, w)
            mse_values.append(val_mse)

        #for all MSEs for this lambda, get mean
        mean_mse = np.mean(mse_values)
        #set best lambda
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_lambda = lambda_

    return best_lambda


def plot_results(lambdas, train_mse_values, test_mse_values, dataset_name):
    """simple plotting function for noncrossvalidation method. Adjust as needed"""
    min_test_mse = min(test_mse_values)
    best_lambda = lambdas[test_mse_values.index(min_test_mse)]

    plt.plot(lambdas, train_mse_values, label='Train MSE', linestyle='dashed')
    plt.plot(lambdas, test_mse_values, label='Test MSE', linestyle='solid')
    plt.axvline(best_lambda, color='r', linestyle='--', label=f'Best Lambda = {best_lambda}')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Train & Test MSE vs Lambda ({dataset_name})')
    plt.legend()
    plt.show()

    print(f'Best λ for {dataset_name}: {best_lambda} with Test MSE: {min_test_mse:.4f}')

def generate_learning_curve(X_train, y_train, X_test, y_test, lambdas, num_repeats=30, step_size=50):
    """learning curve function. Takes subset of training data and finds MSE given a lambda. Repeats 30 times to find an average.
    Repeats for increasing sizes (step size = 50) of the subset of training data."""
    learning_curves = {}
    for lambda_ in lambdas:
        mse_list = []
        for size in range(step_size, X_train.shape[0], step_size):
            mse_for_size = []
            for _ in range(num_repeats):
                #choose random section of data to train
                indices = np.random.choice(X_train.shape[0], size, replace=False)
                X_train_subset = X_train[indices]
                y_train_subset = y_train[indices]

                #do linear regression for the subset
                w = linear_regression(X_train_subset, y_train_subset, lambda_)

                #compute mse for the trained subset
                test_mse = compute_mse(X_test, y_test, w)
                mse_for_size.append(test_mse)

            #average the MSE from the 10 iterations for this size
            learning_curves[lambda_] = np.append(learning_curves.get(lambda_, []), np.mean(mse_for_size))

    return learning_curves

def plotLearningCurve(curve, X):
    """graphing function for the learning curves"""

    plt.figure(figsize=(10, 6))
    for lambda_, mse_values in curve.items():
        training_sizes = np.arange(50, X.shape[0], 50)
        plt.plot(training_sizes, mse_values, label=f"λ={lambda_}")

    plt.title('Learning Curve: Test MSE vs Training Set Size')
    plt.xlabel('Training Set Size')
    plt.ylabel('Test MSE')
    plt.legend()
    plt.grid(True)
    plt.show()