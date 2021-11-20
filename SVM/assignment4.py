from re import T
from SVM.svm import SVM
import numpy as np
from DecisionTree.utilities import extract_data_from_csv


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
    # Primal SVM
    #####################################
    print('Primal SVM')
    print('-------------------------')
    for C in ['(100/873)', '(500/873)', '(700/873)']:
        print(f'Using C value {C} and learning rate function gamma_t = gamma_0 / (1 + (gamma_0 / a) * t)')
        training_data = preprocess_data(extract_data_from_csv('./train.csv'))
        testing_data = preprocess_data(extract_data_from_csv('./test.csv'))

        classifier = SVM(training_data, 100, eval(C), .01, .1)

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
        print(f'Final classifier: {classifier.classifier}')
        print('\n')

        print(f'Using C value {C} and function learning rate function gamma_t = gamma_0 / (1 + t)')
        training_data = preprocess_data(extract_data_from_csv('./train.csv'))
        testing_data = preprocess_data(extract_data_from_csv('./test.csv'))

        classifier = SVM(training_data, 100, eval(C), .01, .1)

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
        print(f'Final classifier: {classifier.classifier}')
        print('\n')
    print('\n\n')

    ####################################
    # Dual SVM
    ####################################
    print('Dual SVM')
    print('-------------------------')
    for C in ['(100/873)', '(500/873)', '(700/873)']:
        print(f'Using C value {C}')
        training_data = preprocess_data(extract_data_from_csv('./train.csv'))
        testing_data = preprocess_data(extract_data_from_csv('./test.csv'))

        classifier = SVM(training_data, None, eval(C), None, None, type="DUAL")

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
        print(f'Final classifier: {classifier.classifier}')
        print('\n')
    print('\n\n')

    print('Dual SVM -- With Gaussian Kernal')
    print('-------------------------')
    for C in ['(100/873)', '(500/873)', '(700/873)']:
        print(f'Using C value {C}')
        training_data = preprocess_data(extract_data_from_csv('./train.csv'))
        testing_data = preprocess_data(extract_data_from_csv('./test.csv'))
        classifier = SVM(training_data, None, eval(C), None, 200, type="GAUSSIAN")

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
        print(f'Final classifier: {classifier.classifier}')
        print('\n')
    print('\n\n')
