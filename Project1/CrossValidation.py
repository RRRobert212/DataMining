#crossvalidation.py

from LinearRegression import *

def main():

    #data is a dictionary with names for keys and tuples of training and test data for values
    datasets = {
        'train-50(1000)-100': ('HW1_dataset/train-50(1000)-100.csv', 'HW1_dataset/test-1000-100.csv'),
        'train-100(1000)-100': ('HW1_dataset/train-100(1000)-100.csv', 'HW1_dataset/test-1000-100.csv'),
        'train-150(1000)-100': ('HW1_dataset/train-150(1000)-100.csv', 'HW1_dataset/test-1000-100.csv'),
        'train-100-10': ('HW1_dataset/train-100-10.csv', 'HW1_dataset/test-100-10.csv'),
        'train-100-100': ('HW1_dataset/train-100-100.csv', 'HW1_dataset/test-100-100.csv'),
        'train-1000-100': ('HW1_dataset/train-1000-100.csv', 'HW1_dataset/test-1000-100.csv')
    }

    #lambda range - 0-150
    lambdas = np.arange(0, 151)


    print("\nCROSS VALIDATION METHOD:\n")
    print("Processing all 6 datasets with λ = 0-150:\n")

    #forloop over the dataset dictionary
    for dataset_name, (train_file, test_file) in datasets.items():

        train_data = loadData(train_file)
        test_data = loadData(test_file)

        #create matrices
        X_train, y_train = createMatrices(train_data)
        X_test, y_test = createMatrices(test_data)

        #find the best lambda using cross validation
        best_lambda = cross_validate(X_train, y_train, lambdas)

        #using the best lambda, train the model with it
        w = linear_regression(X_train, y_train, best_lambda)

        #find the mse on the TEST set
        test_mse = compute_mse(X_test, y_test, w)

        #don't need to plot anything, just print results
        print(f'Best λ for {dataset_name} {best_lambda}: with Test MSE: {test_mse:.4f}')

if __name__ == '__main__':
    main()
