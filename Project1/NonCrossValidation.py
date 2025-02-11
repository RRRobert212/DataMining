from LinearRegression import *

def main():
    #dictionary with train and test as tuples
    datasets = {
        'train-50(1000)-100': ('HW1_dataset/train-50(1000)-100.csv', 'HW1_dataset/test-1000-100.csv'),
        'train-100(1000)-100': ('HW1_dataset/train-100(1000)-100.csv', 'HW1_dataset/test-1000-100.csv'),
        'train-150(1000)-100': ('HW1_dataset/train-150(1000)-100.csv', 'HW1_dataset/test-1000-100.csv'),
        'train-100-10': ('HW1_dataset/train-100-10.csv', 'HW1_dataset/test-100-10.csv'),
        'train-100-100': ('HW1_dataset/train-100-100.csv', 'HW1_dataset/test-100-100.csv'),
        'train-1000-100': ('HW1_dataset/train-1000-100.csv', 'HW1_dataset/test-1000-100.csv')
    }

    #lambda range for the first part (0-150)
    lambdas_0_150 = np.arange(0, 151)

    #process all 6 datasets with λ = 0-150
    print("Processing all 6 datasets with λ = 0-150:")
    for dataset_name, (train_file, test_file) in datasets.items():

        train_data = loadData(train_file)
        test_data = loadData(test_file)


        X_train, y_train = createMatrices(train_data)
        X_test, y_test = createMatrices(test_data)

        #lists to store MSEs
        train_mse_values = []
        test_mse_values = []

        for lambda_ in lambdas_0_150:

            #train it on test set
            w = linear_regression(X_train, y_train, lambda_)

            #compute MSE for train and test, add them to list
            train_mse = compute_mse(X_train, y_train, w)
            test_mse = compute_mse(X_test, y_test, w)
            train_mse_values.append(train_mse)
            test_mse_values.append(test_mse)

        plot_results(lambdas_0_150, train_mse_values, test_mse_values, dataset_name)


    #SECOND PLOTS
    #lambda range for the second part (1-150)
    lambdas_1_150 = np.arange(1, 151)

    print("\nProcessing the first 3 datasets with λ = 1-150:")
    #need to make it a list to get the first 3
    first_three_datasets = list(datasets.items())[:3]
    for dataset_name, (train_file, test_file) in first_three_datasets:

        train_data = loadData(train_file)
        test_data = loadData(test_file)
        X_train, y_train = createMatrices(train_data)
        X_test, y_test = createMatrices(test_data)
        train_mse_values = []
        test_mse_values = []

        for lambda_ in lambdas_1_150:

            w = linear_regression(X_train, y_train, lambda_)


            train_mse = compute_mse(X_train, y_train, w)
            test_mse = compute_mse(X_test, y_test, w)

            train_mse_values.append(train_mse)
            test_mse_values.append(test_mse)
            
        plot_results(lambdas_1_150, train_mse_values, test_mse_values, dataset_name)

if __name__ == '__main__':
    main()