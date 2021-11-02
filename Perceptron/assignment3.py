from Perceptron.perceptron import Perceptron
from DecisionTree.utilities import extract_data_from_csv


def preprocess_data(data):
    numerical_data = []
    for i in range(len(data)):
        row = [1]
        for j in range(len(data[i])):
            row.append(float(data[i][j]))
        numerical_data.append(row)
    return numerical_data

if __name__ == '__main__':
    
    #####################################
    # Standard Perceptron
    #####################################
    print('STANDARD PERCEPTRON')
    print('-------------------------')
    training_data = preprocess_data(extract_data_from_csv('./train.csv'))
    testing_data = preprocess_data(extract_data_from_csv('./test.csv'))

    classifier = Perceptron(training_data, 10)

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
    print('\n\n')

    #####################################
    # Average Perceptron
    #####################################
    print('AVERAGE PERCEPTRON')
    print('-------------------------')
    training_data = preprocess_data(extract_data_from_csv('./train.csv'))
    testing_data = preprocess_data(extract_data_from_csv('./test.csv'))

    classifier = Perceptron(training_data, 10, type='AVERAGE')

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

    print('\n\n')

    #####################################
    # Voted Perceptron
    #####################################
    print('VOTED PERCEPTRON')
    print('-------------------------')
    training_data = preprocess_data(extract_data_from_csv('./train.csv'))
    testing_data = preprocess_data(extract_data_from_csv('./test.csv'))

    classifier = Perceptron(training_data, 10, type='VOTED')

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

    print('Voted Perceptron Classifiers: ')
    for i in range(len(classifier.classifier)):
        print(f'{i}: {classifier.classifier[i][0]}, {classifier.classifier[i][1]}')

    print('\n\n')

