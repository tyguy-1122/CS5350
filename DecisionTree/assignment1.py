import decision_tree
from utilities import extract_data_from_csv
from utilities import calc_classification_error

if __name__ == '__main__':
    training_data = extract_data_from_csv('./car_dataset/train.csv')
    testing_data = extract_data_from_csv('./car_dataset/test.csv')
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

    ENT_errors = []
    for i in range(6):
        DT = decision_tree.DecisionTree(training_data, possible_attribute_values, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'], max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in training_data:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[6] for row in training_data]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in testing_data:
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[6] for row in testing_data]

        testing_error = calc_classification_error(test_predictions, correct_values)
        ENT_errors.append((training_error, testing_error))

    ######################################################
    # Decision Tree Using Majority Error
    ######################################################
    ME_errors = []
    for i in range(6):
        DT = decision_tree.DecisionTree(training_data, possible_attribute_values, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'], method='ME', max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in training_data:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[6] for row in training_data]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in testing_data:
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[6] for row in testing_data]

        testing_error = calc_classification_error(test_predictions, correct_values)
        ME_errors.append((training_error, testing_error))

    ######################################################
    # Decision Tree Using Gini Index
    ######################################################

    GINI_errors = []
    for i in range(6):
        DT = decision_tree.DecisionTree(training_data, possible_attribute_values, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'], method='GINI', max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in training_data:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[6] for row in training_data]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in testing_data:
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[6] for row in testing_data]

        testing_error = calc_classification_error(test_predictions, correct_values)
        GINI_errors.append((training_error, testing_error))

    

    ######################################################
    # Print out a table of errors
    ######################################################
    print('--- Question 2 - Basic ID3 Algorithm Implementation ---')
    print('Algorithm\t', 'type\t\t', '1\t', '2\t', '3\t', '4\t', '5\t', '6\t')
    print('\n')
    print('Entropy\t\t', 'training\t', round(ENT_errors[0][0], 2), '\t', round(ENT_errors[1][0], 2), '\t', round(ENT_errors[2][0], 2), '\t', round(ENT_errors[3][0], 2), '\t', round(ENT_errors[4][0], 2), '\t', round(ENT_errors[5][0], 2))
    print('\n')
    print('Entropy\t\t', 'testing\t', round(ENT_errors[0][1], 2), '\t', round(ENT_errors[1][1], 2), '\t', round(ENT_errors[2][1], 2), '\t', round(ENT_errors[3][1], 2), '\t', round(ENT_errors[4][1], 2), '\t', round(ENT_errors[5][1], 2))
    print('\n')
    print('Majority Error\t', 'training\t', round(ME_errors[0][0], 2), '\t', round(ME_errors[1][0], 2), '\t', round(ME_errors[2][0], 2), '\t', round(ME_errors[3][0], 2), '\t', round(ME_errors[4][0], 2), '\t', round(ME_errors[5][0], 2))
    print('\n')
    print('Majority Error\t', 'testing\t', round(ME_errors[0][1], 2), '\t', round(ME_errors[1][1], 2), '\t', round(ME_errors[2][1], 2), '\t', round(ME_errors[3][1], 2), '\t', round(ME_errors[4][1], 2), '\t', round(ME_errors[5][1], 2))
    print('\n')
    print('Gini Index\t', 'training\t', round(GINI_errors[0][0], 2), '\t', round(GINI_errors[1][0], 2), '\t', round(GINI_errors[2][0], 2), '\t', round(GINI_errors[3][0], 2), '\t', round(GINI_errors[4][0], 2), '\t', round(GINI_errors[5][0], 2))
    print('\n')
    print('Gini Index\t', 'testing\t', round(GINI_errors[0][1], 2), '\t', round(GINI_errors[1][1], 2), '\t', round(GINI_errors[2][1], 2), '\t', round(GINI_errors[3][1], 2), '\t', round(GINI_errors[4][1], 2), '\t', round(GINI_errors[5][1], 2))









    ######################################################
    ######################################################
    # Question 2
    ######################################################
    ######################################################


    print('\n\n\n')

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

    ######################################################
    ######################################################
    # Preprocessing --- Numerical Attributes Only
    ######################################################
    ######################################################

    print('\n\n')
    ######################################################
    # Decision Tree Using Entropy
    ######################################################

    ENT_errors = []
    for i in range(16):
        bank_data_training = extract_data_from_csv('./bank/train.csv')
        bank_data_testing = extract_data_from_csv('./bank/test.csv')
        DT = decision_tree.DecisionTree(bank_data_training, possible_attribute_values_bank_with_unknown,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
        'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], numerical_vals=True, max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in bank_data_training:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_training]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in decision_tree.DecisionTree.convert_numerical_to_binary(DT, bank_data_testing):
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_testing]

        testing_error = calc_classification_error(test_predictions, correct_values)
        ENT_errors.append((training_error, testing_error))

    ######################################################
    # Decision Tree Using Majority Error
    ######################################################
    ME_errors = []
    for i in range(16):
        bank_data_training = extract_data_from_csv('./bank/train.csv')
        bank_data_testing = extract_data_from_csv('./bank/test.csv')
        DT = decision_tree.DecisionTree(bank_data_training, possible_attribute_values_bank_with_unknown,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
        'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], numerical_vals=True, method='ME', max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in bank_data_training:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_training]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in decision_tree.DecisionTree.convert_numerical_to_binary(DT, bank_data_testing):
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_testing]

        testing_error = calc_classification_error(test_predictions, correct_values)
        ME_errors.append((training_error, testing_error))

    ######################################################
    # Decision Tree Using Gini Index
    ######################################################
    GINI_errors = []
    for i in range(16):
        bank_data_training = extract_data_from_csv('./bank/train.csv')
        bank_data_testing = extract_data_from_csv('./bank/test.csv')
        DT = decision_tree.DecisionTree(bank_data_training, possible_attribute_values_bank_with_unknown,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
        'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], numerical_vals=True, method='GINI', max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in bank_data_training:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_training]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in decision_tree.DecisionTree.convert_numerical_to_binary(DT, bank_data_testing):
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_testing]

        testing_error = calc_classification_error(test_predictions, correct_values)
        GINI_errors.append((training_error, testing_error))



    ######################################################
    # Print out a table of errors
    ######################################################
    print('--- Question 3 - Make Numerical Attributes Binary Only ---')
    print('Algorithm\t', 'type\t\t', end='')
    for i in range(16):
        print(i+1, '\t', end='')
    print('\n')

    print('Entropy\t\t', 'training\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][0], 2), '\t', end='')
    print('\n')
    print('Entropy\t\t', 'testing\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][1], 2), '\t', end='')
    print('\n')


    print('Majority Error\t', 'training\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][0], 2), '\t', end='')
    print('\n')

    print('Majority Error\t', 'testing\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][1], 2), '\t', end='')
    print('\n')

    print('Gini Index\t', 'training\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][0], 2), '\t', end='')
    print('\n')

    print('Gini Index\t', 'testing\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][1], 2), '\t', end='')
    print('\n')







    ######################################################################
    ######################################################################
    # Preprocessing --- Numerical Attributes and Unknown Value Handling
    ######################################################################
    ######################################################################

    print('\n\n')
    ######################################################
    # Decision Tree Using Entropy
    ######################################################

    ENT_errors = []
    for i in range(16):
        bank_data_training = extract_data_from_csv('./bank/train.csv')
        bank_data_testing = extract_data_from_csv('./bank/test.csv')
        DT = decision_tree.DecisionTree(bank_data_training, possible_attribute_values_bank,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
        'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], numerical_vals=True, unknown_vals=True, max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in bank_data_training:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_training]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in decision_tree.DecisionTree.handle_missing_attributes(DT, decision_tree.DecisionTree.convert_numerical_to_binary(DT, bank_data_testing)):
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_testing]

        testing_error = calc_classification_error(test_predictions, correct_values)
        ENT_errors.append((training_error, testing_error))

    ######################################################
    # Decision Tree Using Majority Error
    ######################################################
    ME_errors = []
    for i in range(16):
        bank_data_training = extract_data_from_csv('./bank/train.csv')
        bank_data_testing = extract_data_from_csv('./bank/test.csv')
        DT = decision_tree.DecisionTree(bank_data_training, possible_attribute_values_bank,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
        'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], numerical_vals=True, unknown_vals=True, method='ME', max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in bank_data_training:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_training]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in decision_tree.DecisionTree.handle_missing_attributes(DT, decision_tree.DecisionTree.convert_numerical_to_binary(DT, bank_data_testing)):
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_testing]

        testing_error = calc_classification_error(test_predictions, correct_values)
        ME_errors.append((training_error, testing_error))

    ######################################################
    # Decision Tree Using Gini Index
    ######################################################
    GINI_errors = []
    for i in range(16):
        bank_data_training = extract_data_from_csv('./bank/train.csv')
        bank_data_testing = extract_data_from_csv('./bank/test.csv')
        DT = decision_tree.DecisionTree(bank_data_training, possible_attribute_values_bank,
        ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
        'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], numerical_vals=True, unknown_vals=True, method='GINI', max_depth=i+1)

        # Get error for training data
        train_predictions = []
        for row in bank_data_training:
            train_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_training]

        training_error = calc_classification_error(train_predictions, correct_values)

        # Get error for test data
        test_predictions = []
        for row in decision_tree.DecisionTree.handle_missing_attributes(DT, decision_tree.DecisionTree.convert_numerical_to_binary(DT, bank_data_testing)):
            test_predictions.append(DT.classify_data(row))
        correct_values = [row[-1] for row in bank_data_testing]

        testing_error = calc_classification_error(test_predictions, correct_values)
        GINI_errors.append((training_error, testing_error))


    ######################################################
    # Print out a table of errors
    ######################################################
    print('--- Question 3 - Handle Numerical Attributes and Unknown Values ---')
    print('Algorithm\t', 'type\t\t', end='')
    for i in range(16):
        print(i+1, '\t', end='')
    print('\n')

    print('Entropy\t\t', 'training\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][0], 2), '\t', end='')
    print('\n')
    print('Entropy\t\t', 'testing\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][1], 2), '\t', end='')
    print('\n')


    print('Majority Error\t', 'training\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][0], 2), '\t', end='')
    print('\n')

    print('Majority Error\t', 'testing\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][1], 2), '\t', end='')
    print('\n')

    print('Gini Index\t', 'training\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][0], 2), '\t', end='')
    print('\n')

    print('Gini Index\t', 'testing\t', end='')
    for i in range(16):
        print(round(ENT_errors[i][1], 2), '\t', end='')
    print('\n')
