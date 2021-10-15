from DecisionTree.utilities import extract_data_from_csv
from DecisionTree.utilities import calc_classification_error
from DecisionTree.decision_tree import DecisionTree
from EnsembleLearning.ada_boost import AdaBoost
from EnsembleLearning.bagging import Bagging
from EnsembleLearning.random_forest import RandomForest

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
    for i in range(30,31,1):
        # Read data from .csv for each iteration
        bank_data_training = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/train.csv')
        bank_data_testing = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/test.csv')

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
        #print(classification_error)

        # Calculate the training error
        true_labels = [row[-1] for row in bank_data_training]
        predicted_labels = []
        for row in bank_data_training:
            predicted_labels.append(classifier.classify_data(row, label_map))
        
        training_error = calc_classification_error(predicted_labels, true_labels)
        print(training_error)

    #############################################
    # Problem 2 - B
    #############################################

    for i in range(1, 500, 20):
        # Read data from .csv for each iteration
        bank_data_training = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/train.csv')
        bank_data_testing = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/test.csv')

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
        print(classification_error)

    #############################################
    # Problem 2 - D
    #############################################

    for i in range(1, 500, 20):
        # Read data from .csv for each iteration
        bank_data_training = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/train.csv')
        bank_data_testing = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/test.csv')

        # Get the correct labels for the testing data for calculating classification error
        true_labels = [row[-1] for row in bank_data_testing]

        # Build the classifier with the specified number of weak classifiers
        classifier = RandomForest(bank_data_training, possible_attribute_values_bank_with_unknown,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
            'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], i, 5)
        
        # Convert the numerical data in test data to binary
        DecisionTree.convert_numerical_to_binary(classifier.weak_classifiers[0], bank_data_testing)

        # Get the predicted labels
        predicted_labels = []
        for row in bank_data_testing:
            predicted_labels.append(classifier.classify_data(row))
        
        # Calculate the classification error
        classification_error = calc_classification_error(predicted_labels, true_labels)
        print(classification_error)

 