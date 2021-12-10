from NeuralNetworks.neural_net import NeuralNet
from DecisionTree.utilities import extract_data_from_csv
import numpy as np
import torch
from torch.nn import nn
from torch.optim import Adam
import copy

def preprocess_data(data):
    numerical_data = []
    for i in range(len(data)):
        row = [1]
        for j in range(len(data[i])):
            row.append(float(data[i][j]))
        numerical_data.append(row)

    for row in numerical_data:
        if row[-1] == 0:
            row[-1] = -1

    return numerical_data

if __name__ == '__main__':
    #####################################
    # Neural Networks - My Implementation
    #####################################
    print('Neural Net - Weights at 0')
    print('-------------------------')
    for hidden_neurons in [5, 10, 25, 50, 100]:
    # for hidden_neurons in [65]:
        print(f'Using {hidden_neurons} hidden neurons in the hidden layers')
        training_data = preprocess_data(extract_data_from_csv('./train.csv'))
        testing_data = preprocess_data(extract_data_from_csv('./test.csv'))

        # classifier = NeuralNet(training_data, 10, hidden_neurons=hidden_neurons)
        classifier = NeuralNet(training_data, 10, hidden_neurons=hidden_neurons)

        true_test_labels = [row[-1] for row in testing_data]
        true_train_labels = [row[-1] for row in training_data]

        # Calculate test error
        test_predictions = []
        for row in testing_data:
            test_predictions.append(classifier.classify_data(row))
        
        incorrect_test_labels = 0
        for i in range(len(true_test_labels)):
            if true_test_labels[i] != test_predictions[i]: incorrect_test_labels += 1
        
        print('Testing Error: ', incorrect_test_labels / len(true_test_labels))

        # Calculate training error
        train_predictions = []
        for row in training_data:
            train_predictions.append(classifier.classify_data(row))
        
        incorrect_train_labels = 0
        for i in range(len(true_train_labels)):
            if true_train_labels[i] != train_predictions[i]: incorrect_train_labels += 1
        
        print('Training Error: ', incorrect_train_labels / len(true_train_labels))
        print('\n')

    print('\n\n')

    # #####################################
    # # Neural Networks - Pytorch
    # #####################################
    # print('Neural Net - Pytorch Version')
    # print('-------------------------')
    # for hidden_neurons in [5, 10, 25, 50, 100]:
    # # for hidden_neurons in [65]:
    #     print(f'Using {hidden_neurons} hidden neurons in the hidden layers')
    #     training_data = preprocess_data(extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/NeuralNetworks/train.csv'))
    #     testing_data = preprocess_data(extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/NeuralNetworks/test.csv'))

    #     true_test_labels = [row[-1] for row in testing_data]
    #     true_train_labels = [row[-1] for row in training_data]

    #     # Calculate test error
    #     test_predictions = []
    #     for row in testing_data:
    #         test_predictions.append(classifier.classify_data(row))
        
    #     incorrect_test_labels = 0
    #     for i in range(len(true_test_labels)):
    #         if true_test_labels[i] != test_predictions[i]: incorrect_test_labels += 1
        
    #     print('Testing Error: ', incorrect_test_labels / len(true_test_labels))

    #     # Calculate training error
    #     train_predictions = []
    #     for row in training_data:
    #         train_predictions.append(classifier.classify_data(row))
        
    #     incorrect_train_labels = 0
    #     for i in range(len(true_train_labels)):
    #         if true_train_labels[i] != train_predictions[i]: incorrect_train_labels += 1
        
    #     print('Training Error: ', incorrect_train_labels / len(true_train_labels))
    #     print('\n')

    print('\n\n')