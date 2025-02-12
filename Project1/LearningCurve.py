from LinearRegression import *

def main():

    datasets = {
        'train-1000-100': ('HW1_dataset/train-1000-100.csv', 'HW1_dataset/test-1000-100.csv')
    }

    lambdas = [1, 25, 150]

    train_data = loadData(datasets['train-1000-100'][0])
    test_data = loadData(datasets['train-1000-100'][1])

    X_train, y_train = createMatrices(train_data)
    X_test, y_test = createMatrices(test_data)

    learning_curves = generate_learning_curve(X_train, y_train, X_test, y_test, lambdas)

    plotLearningCurve(learning_curves, X_train)

if __name__ == '__main__':
    main()
