import decision_tree
from utilities import extract_data_from_csv
from utilities import calc_classification_error

if __name__ == '__main__':
    training_data = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/car_dataset/train.csv')
    testing_data = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/car_dataset/test.csv')
    possible_attribute_values = {
        'buying':   ['vhigh', 'high', 'med', 'low'],
        'maint':    ['vhigh', 'high', 'med', 'low'],
        'doors':    ['2', '3', '4', '5more'],
        'persons':  ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety':   ['low', 'med', 'high']
    }
    ######################################################
    ######################################################
    # Question 2
    ######################################################
    ######################################################

    print('\n\n')
    ######################################################
    # Decision Tree Using Entropy
    ######################################################

    DT = decision_tree.DecisionTree(training_data, possible_attribute_values, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])

    # Get error for training data
    train_predictions = []
    for row in training_data:
        train_predictions.append(DT.classify_data(row))
    correct_values = [row[6] for row in training_data]

    print('---Decision Tree with Entropy Equation---')
    print('training error: ', calc_classification_error(train_predictions, correct_values))

    # Get error for test data
    test_predictions = []
    for row in testing_data:
        test_predictions.append(DT.classify_data(row))
    correct_values = [row[6] for row in testing_data]

    print('testing error: ', calc_classification_error(test_predictions, correct_values))
    print('\n\n')

    ######################################################
    # Decision Tree Using Majority Error
    ######################################################

    DT = decision_tree.DecisionTree(training_data, possible_attribute_values, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'], method='ME')

    # Get error for training data
    train_predictions = []
    for row in training_data:
        train_predictions.append(DT.classify_data(row))
    correct_values = [row[6] for row in training_data]

    print('---Decision Tree with Majority Error Equation---')
    print('training error: ', calc_classification_error(train_predictions, correct_values))

    # Get error for test data
    test_predictions = []
    for row in testing_data:
        test_predictions.append(DT.classify_data(row))
    correct_values = [row[6] for row in testing_data]

    print('testing error: ', calc_classification_error(test_predictions, correct_values))
    print('\n\n')

    ######################################################
    # Decision Tree Using Gini Index
    ######################################################

    DT = decision_tree.DecisionTree(training_data, possible_attribute_values, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'], method='GINI')

    # Get error for training data
    train_predictions = []
    for row in training_data:
        train_predictions.append(DT.classify_data(row))
    correct_values = [row[6] for row in training_data]

    print('---Decision Tree with Gini Index Equation---')
    print('training error: ', calc_classification_error(train_predictions, correct_values))

    # Get error for test data
    test_predictions = []
    for row in testing_data:
        test_predictions.append(DT.classify_data(row))
    correct_values = [row[6] for row in testing_data]

    print('testing error: ', calc_classification_error(test_predictions, correct_values))
    print('\n\n')



    ## Majority Error test
    me_test_data = [
        ['s','h','h','w','-'],
        ['s','h','h','s','-'],
        ['o','h','h','w','+'],
        ['r','m','h','w','+'],
        ['s','m','h','w','-'],
        ['o','m','h','s','+'],
        ['r','m','h','s','-'],
    ]
    print(decision_tree.calc_information_gain(me_test_data, [0,1,3], 'ME'))