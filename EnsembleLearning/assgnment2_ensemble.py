from DecisionTree.utilities import extract_data_from_csv
from DecisionTree.utilities import calc_classification_error
from DecisionTree.decision_tree import DecisionTree
from EnsembleLearning.ada_boost import AdaBoost
from EnsembleLearning.bagging import Bagging
from EnsembleLearning.random_forest import RandomForest
import matplotlib.pyplot as plt
from statistics import mean, variance, median

if __name__ == '__main__':

    #############################################
    #############################################
    # Problem 2 - Ensemble Learning
    #############################################
    #############################################

    possible_attribute_values_bank = {
    'age': ['-','+'],
    'job': ['admin.','unemployed','management','housemaid','entrepreneur','student',
        'blue-collar','self-employed','retired','technician','services'],
    'marital' : ['married','divorced','single'],
    'education': ['secondary','primary','tertiary'],
    'default': ['yes','no'],
    'balance': ['-','+'],
    'housing': ['yes','no'],
    'loan': ['yes','no'],
    'contact': ['telephone','cellular'],
    'day': ['-','+'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'duration': ['-','+'],
    'campaign': ['-','+'],
    'pdays': ['-','+'],
    'previous': ['-','+'],
    'poutcome': ['other','failure','success']
    }

    possible_attribute_values_bank_with_unknown = {
    'age': ['-','+'],
    'job': ['admin.','unknown','unemployed','management','housemaid','entrepreneur','student',
        'blue-collar','self-employed','retired','technician','services'],
    'marital' : ['married','divorced','single'],
    'education': ['unknown','secondary','primary','tertiary'],
    'default': ['yes','no'],
    'balance': ['-','+'],
    'housing': ['yes','no'],
    'loan': ['yes','no'],
    'contact': ['unknown','telephone','cellular'],
    'day': ['-','+'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'duration': ['-','+'],
    'campaign': ['-','+'],
    'pdays': ['-','+'],
    'previous': ['-','+'],
    'poutcome': ['unknown','other','failure','success']
    }

    #############################################
    # Problem 2 - A
    #############################################
    label_map = {
        'no': -1,
        'yes': 1,
        'negative': 'no',
        'positive': 'yes'
    }
    adaboost_sizes = [1,5,10,15,20,50,100,250,500]
    classification_errors = []
    training_errors = []
    weak_classifiers_end = []
    classifier = None
    for i in adaboost_sizes:
        print(f'Working on AdaBoost with {i} weak classifiers')
        # Read data from .csv for each iteration
        bank_data_training = extract_data_from_csv('../DecisionTree/bank/train.csv')
        bank_data_testing = extract_data_from_csv('../DecisionTree/bank/test.csv')

        # Add a placeholder for the weights column to avoid index issues
        for row in bank_data_testing:
            row.insert(0, 'placeholder')

        # Get the correct labels for the testing data for calculating classification error
        true_labels = [row[-1] for row in bank_data_testing]

        # Build the classifier with the specified number of weak classifiers
        classifier = AdaBoost(bank_data_training, possible_attribute_values_bank_with_unknown,
        ['weights', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
            'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], i)
        
        # Convert the numerical data in test data to binary
        DecisionTree.convert_numerical_to_binary(classifier.weak_classifiers[0], bank_data_testing)

        # Get the predicted labels
        predicted_labels = []
        for row in bank_data_testing:
            predicted_labels.append(classifier.classify_data(row, label_map))
        
        # Calculate the classification error
        classification_error = calc_classification_error(predicted_labels, true_labels)
        
        # Calculate the training error
        true_labels = [row[-1] for row in bank_data_training]
        predicted_labels = []
        for row in bank_data_training:
            predicted_labels.append(classifier.classify_data(row, label_map))
        
        training_error = calc_classification_error(predicted_labels, true_labels)

        classification_errors.append(classification_error)
        training_errors.append(training_error)
        weak_classifiers_end = classifier.weak_classifiers
    
    print('\n\n\n----------------------\n\n\n')

    print('Testing Errors')
    for error in classification_errors:
        print(error)
    
    print('\n\n\n----------------------\n\n\n')

    print('Training Errors')
    for error in training_errors:
        print(error)

    print('\n\n\n----------------------\n\n\n')



    bank_data_training = extract_data_from_csv('../DecisionTree/bank/train.csv')
    bank_data_testing = extract_data_from_csv('../DecisionTree/bank/test.csv')
    for row in bank_data_training:
        row.insert(0,0)
    for row in bank_data_testing:
        row.insert(0,0)
    DecisionTree.convert_numerical_to_binary(classifier.weak_classifiers[0], bank_data_testing)
    DecisionTree.convert_numerical_to_binary(classifier.weak_classifiers[0], bank_data_training)

    testing_error_indiv = []
    testing_correct = [row[-1] for row in bank_data_testing]
    training_error_indiv = []
    training_correct = [row[-1] for row in bank_data_training]
    for weak_classifier in weak_classifiers_end:
        predicted_labels_testing = []
        predicted_labels_training = []  
        for row in bank_data_testing:
            predicted_labels_testing.append(weak_classifier.classify_data(row))
        
        for row in bank_data_training:
            predicted_labels_training.append(weak_classifier.classify_data(row))

        # Calculate the classification error
        testing_error_indiv.append(calc_classification_error(predicted_labels_testing, testing_correct))
        training_error_indiv.append(calc_classification_error(predicted_labels_training, training_correct))
        

    print('Testing error for individual weak classifiers')
    print(testing_error_indiv)

    print('\n\n\n----------------------\n\n\n')

    print('Training error for individual weak classifiers')
    print(training_error_indiv)

    #############################################
    # Problem 2 - B
    #############################################

    bagging_sizes = [1,5,10,15,20,50,100,250,500]
    classification_errors = []
    training_errors = []
    weak_classifiers_end = []
    for i in bagging_sizes:
        print(f'Working on Bagging with {i} weak classifiers')
        # Read data from .csv for each iteration
        bank_data_training = extract_data_from_csv('../DecisionTree/bank/train.csv')
        bank_data_testing = extract_data_from_csv('../DecisionTree/bank/test.csv')

        # Get the correct labels for the testing data for calculating classification error
        true_labels = [row[-1] for row in bank_data_testing]

        # Build the classifier with the specified number of weak classifiers
        classifier = Bagging(bank_data_training, possible_attribute_values_bank_with_unknown,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
            'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], i)
        
        # Convert the numerical data in test data to binary
        DecisionTree.convert_numerical_to_binary(classifier.weak_classifiers[0], bank_data_testing)

        # Get the predicted labels
        predicted_labels = []
        for row in bank_data_testing:
            predicted_labels.append(classifier.classify_data(row))
        
        # Calculate the classification error
        classification_error = calc_classification_error(predicted_labels, true_labels)
             
        # Calculate the training error
        true_labels = [row[-1] for row in bank_data_training]
        predicted_labels = []
        for row in bank_data_training:
            predicted_labels.append(classifier.classify_data(row))
        
        training_error = calc_classification_error(predicted_labels, true_labels)

        classification_errors.append(classification_error)
        training_errors.append(training_error)

    print('\n\n\n----------------------\n\n\n')

    print('Testing Errors')
    for error in classification_errors:
        print(error)
    
    print('\n\n\n----------------------\n\n\n')

    print('Training Errors')
    for error in training_errors:
        print(error)

    print('\n\n\n----------------------\n\n\n')

    #############################################
    # Problem 2 - D
    #############################################

    random_forest_sizes = [1,5,10,15,20,50,100,250,500]
    classification_errors = []
    training_errors = []
    weak_classifiers_end = []
    for i in random_forest_sizes:
        print(f'Working on Random Forest with {i} weak classifiers and subsets of size 6')
        # Read data from .csv for each iteration
        bank_data_training = extract_data_from_csv('../DecisionTree/bank/train.csv')
        bank_data_testing = extract_data_from_csv('../DecisionTree/bank/test.csv')

        # Get the correct labels for the testing data for calculating classification error
        true_labels = [row[-1] for row in bank_data_testing]

        # Build the classifier with the specified number of weak classifiers
        classifier = RandomForest(bank_data_training, possible_attribute_values_bank_with_unknown,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
            'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], i, 6)
        
        # Convert the numerical data in test data to binary
        DecisionTree.convert_numerical_to_binary(classifier.weak_classifiers[0], bank_data_testing)

        # Get the predicted labels
        predicted_labels = []
        for row in bank_data_testing:
            predicted_labels.append(classifier.classify_data(row))
        
        # Calculate the classification error
        classification_error = calc_classification_error(predicted_labels, true_labels)

        # Calculate the training error
        true_labels = [row[-1] for row in bank_data_training]
        predicted_labels = []
        for row in bank_data_training:
            predicted_labels.append(classifier.classify_data(row))
        
        training_error = calc_classification_error(predicted_labels, true_labels)

        classification_errors.append(classification_error)
        training_errors.append(training_error)

    print('\n\n\n----------------------\n\n\n')

    print('Testing Errors')
    for error in classification_errors:
        print(error)
    
    print('\n\n\n----------------------\n\n\n')

    print('Training Errors')
    for error in training_errors:
        print(error)

    print('\n\n\n----------------------\n\n\n')


    #############################################
    # Problem 2 - C
    #############################################
    classifiers = []
    column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
            'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    bank_data_training = extract_data_from_csv('../DecisionTree/bank/train.csv')
    bank_data_testing = extract_data_from_csv('../DecisionTree/bank/test.csv')
    # Convert data to binary from numerical
    for i in range(len(bank_data_testing[0]) - 1):
        if ['-', '+'] == possible_attribute_values_bank[column_headers[i]] and bank_data_testing[0][i] not in ['-', '+']: # Column is numerical and not adjusted
            # Find the median value
            median_val = median([int(row[i]) for row in bank_data_testing])
            # Replace all numbers with either + or -
            for row in bank_data_testing:
                if int(row[i]) >= median_val:
                    row[i] = '+'
                else:
                    row[i] = '-'
    for i in range(30):
        print(f'Making classifier number {i}')
        classifier = Bagging(bank_data_training, possible_attribute_values_bank_with_unknown, column_headers, 20, num_samples=1000)
        classifiers.append(classifier)
    
    # Get the general squared error for single trees
    bias_per_example = []
    variance_per_example = []
    single_trees = [classifier.weak_classifiers[0] for classifier in classifiers]
    for row in bank_data_testing:
        ground_truth_label = 1 if row[-1] == 'yes' else -1
        predictions_per_example = []
        for tree in single_trees:
            label = tree.classify_data(row)
            predictions_per_example.append(1 if label == 'yes' else -1)
        average = mean(predictions_per_example)
        bias = (average - ground_truth_label) ** 2
        bias_per_example.append(bias)
        var = variance(predictions_per_example)
        variance_per_example.append(var)
    mean_bias_single_trees = mean(bias_per_example)
    mean_var_single_trees = mean(variance_per_example)

    print('-------------------')
    print(f'The bias for single trees is {mean_bias_single_trees}')
    print(f'The variance for single trees is {mean_var_single_trees}')
    print(f'The General Squared Error for single trees is {mean_bias_single_trees+mean_var_single_trees}')
    print('-------------------')


    # Get the general squared error for bagged classifiers
    bias_per_example = []
    variance_per_example = []
    for row in bank_data_testing:
        ground_truth_label = 1 if row[-1] == 'yes' else -1
        predictions_per_example = []
        for classifier in classifiers:
            label = classifier.classify_data(row)
            predictions_per_example.append(1 if label == 'yes' else -1)
        average = mean(predictions_per_example)
        bias = (average - ground_truth_label) ** 2
        bias_per_example.append(bias)
        var = variance(predictions_per_example)
        variance_per_example.append(var)
    mean_bias = mean(bias_per_example)
    mean_var = mean(variance_per_example)

    print('-------------------')
    print(f'The bias for full bagged classifiers is {mean_bias}')
    print(f'The variance for full bagged classifiers is {mean_var}')
    print(f'The General Squared Error for full bagged classifiers is {mean_bias+mean_var}')
    print('-------------------')














    classifiers = []
    column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
            'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    bank_data_training = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/train.csv')
    bank_data_testing = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/test.csv')
    # Convert data to binary from numerical
    for i in range(len(bank_data_testing[0]) - 1):
        if ['-', '+'] == possible_attribute_values_bank[column_headers[i]] and bank_data_testing[0][i] not in ['-', '+']: # Column is numerical and not adjusted
            # Find the median value
            median_val = median([int(row[i]) for row in bank_data_testing])
            # Replace all numbers with either + or -
            for row in bank_data_testing:
                if int(row[i]) >= median_val:
                    row[i] = '+'
                else:
                    row[i] = '-'
    for i in range(30):
        print(f'Making classifier number {i}')
        classifier = RandomForest(bank_data_training, possible_attribute_values_bank_with_unknown, column_headers, 20, 6)
        classifiers.append(classifier)
    
    # Get the general squared error for single trees
    bias_per_example = []
    variance_per_example = []
    single_trees = [classifier.weak_classifiers[0] for classifier in classifiers]
    for row in bank_data_testing:
        ground_truth_label = 1 if row[-1] == 'yes' else -1
        predictions_per_example = []
        for tree in single_trees:
            label = tree.classify_data(row)
            predictions_per_example.append(1 if label == 'yes' else -1)
        average = mean(predictions_per_example)
        bias = (average - ground_truth_label) ** 2
        bias_per_example.append(bias)
        var = variance(predictions_per_example)
        variance_per_example.append(var)
    mean_bias_single_trees = mean(bias_per_example)
    mean_var_single_trees = mean(variance_per_example)

    print('-------------------')
    print(f'The bias for single random trees is {mean_bias_single_trees}')
    print(f'The variance for single random trees is {mean_var_single_trees}')
    print(f'The General Squared Error for single random trees is {mean_bias_single_trees+mean_var_single_trees}')
    print('-------------------')


    # Get the general squared error for bagged classifiers
    bias_per_example = []
    variance_per_example = []
    for row in bank_data_testing:
        ground_truth_label = 1 if row[-1] == 'yes' else -1
        predictions_per_example = []
        for classifier in classifiers:
            label = classifier.classify_data(row)
            predictions_per_example.append(1 if label == 'yes' else -1)
        average = mean(predictions_per_example)
        bias = (average - ground_truth_label) ** 2
        bias_per_example.append(bias)
        var = variance(predictions_per_example)
        variance_per_example.append(var)
    mean_bias = mean(bias_per_example)
    mean_var = mean(variance_per_example)

    print('-------------------')
    print(f'The bias for full random forests is {mean_bias}')
    print(f'The variance for full random forests is {mean_var}')
    print(f'The General Squared Error for full random forests is {mean_bias+mean_var}')
    print('-------------------')
    