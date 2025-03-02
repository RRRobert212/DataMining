#DataMining HW2#
#k-nearest neighbor classification algorithms#

#-------------------------------------------#

import csv
import math
from collections import Counter
import numpy as np

#-------------------------------------------#

#load the file, ignore the header and first column
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = []
        for row in reader:

            data.append([float(value.strip()) for value in row[1:]])  #Skip the first column

    return data

#normalize the data, uses numpy, returns it as a list
def z_score_normalize(data):
    data = np.array(data)
    #skip last column
    features = data[:, :-1]
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    #normalization function
    features_normalized = (features - mean) / std
    data[:, :-1] = features_normalized
    return data.tolist()

#standard euclidean distance function to determine nearest neighbor
def euclidean_distance(row1, row2):

    return math.sqrt(sum((a - b)**2 for a, b in zip(row1[:-1], row2[:-1])))


#gets distances of the neightbors
def get_neighbors(training_data, test_row, k):
    distances = [(train_row, euclidean_distance(test_row, train_row)) for train_row in training_data]
    distances.sort(key=lambda x: x[1])
    return [row[0] for row in distances[:k]]

#uses majority vote (most_common built-in) to classify based on class of neighbors
def predict_classification(training_data, test_row, k):
    neighbors = get_neighbors(training_data, test_row, k)
    classes = [row[-1] for row in neighbors]
    return Counter(classes).most_common(1)[0][0]

#knn classification without normalization
def knn_classifier(train_file, test_file, k):
    train_data = load_csv(train_file)
    test_data = load_csv(test_file)
    
    predictions = [predict_classification(train_data, test_row, k) for test_row in test_data]
    accuracy = calculate_accuracy(test_data, predictions)
    
    return accuracy

#part C, prints predictions for first 50
def generate_predictions(train_file, test_file, k_values):
    train_data = load_csv(train_file)
    test_data = load_csv(test_file)
    
    #normalize it
    train_data_normalized = z_score_normalize(train_data)
    test_data_normalized = z_score_normalize(test_data)
    
    first_50= test_data_normalized[:50]
    
    #store predictions in a dict
    predictions = {f"t{i+1}": [] for i in range(50)}
    
    #predict spam or not spam and add strings to the list
    for k in k_values:
        for i, test_row in enumerate(first_50):
            predicted_label = predict_classification(train_data_normalized, test_row, k)
            predictions[f"t{i+1}"].append("spam" if predicted_label == 1 else "no")
    
    return predictions


#determines the accuracy based on the test data, simple number of correct predicts over total
def calculate_accuracy(test_data, predictions):
    correct = 0
    for i in range(len(test_data)):
        if test_data[i][-1] == predictions[i]:
            correct += 1
    return correct / len(test_data)

#same as knn_classifier but normalized, copy-pasted func but makes mains simpler
def knn_classifier_normalized(train_file, test_file, k):
    train_data = load_csv(train_file)
    test_data = load_csv(test_file)
    
    #normalize!
    train_data_normalized = z_score_normalize(train_data)
    test_data_normalized = z_score_normalize(test_data)
    
    predictions = [predict_classification(train_data_normalized, test_row, k) for test_row in test_data_normalized]
    accuracy = calculate_accuracy(test_data_normalized, predictions)
    
    return accuracy

#-------------------------------------------#

###MAIN FUNCITONS###
##separated into 3 mains, similar, but correspond directly questions 1.A, 1.B, 1.C###

def mainA():
    train_file = "HW2_dataset/spam_train.csv"
    test_file = "HW2_dataset/spam_test.csv"
    k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
    
    for k in k_values:
        accuracy = knn_classifier(train_file, test_file, k)
        print(f"k = {k}: Test Accuracy = {accuracy:.4f}")


def mainB():
    train_file = "HW2_dataset/spam_train.csv"
    test_file = "HW2_dataset/spam_test.csv"
    k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
    
    for k in k_values:
        accuracy = knn_classifier_normalized(train_file, test_file, k)
        print(f"k = {k}: Test Accuracy = {accuracy:.4f}")

def mainC():
    train_file = "HW2_dataset/spam_train.csv"
    test_file = "HW2_dataset/spam_test.csv"
    k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
    
    predictions = generate_predictions(train_file, test_file, k_values)
    
    for instance, labels in predictions.items():
        print(f"{instance} {', '.join(labels)}")


#-------------------------------------------#

if __name__ == "__main__":
    mainA() #CHANGE THIS TO mainA() mainB() or mainC() DEPENDING ON WHICH YOU WANT TO RUN