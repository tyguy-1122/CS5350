This is a machine learning library developed by Tyler Jones for CS5350 at the University of Utah

## Each top-level assignment folder contains a run.sh script. Use `bash run.sh` within that directory to execute the assignment .py file within that directory. This works on the CADE machines.

How to use DecisionTree:
----------------------------------
1. Create a DecisionTree object
    Parameters:
    - data: The training data set
    - possible_attribute_values: a diction of indexes mapped to a list of the possible attribute values of
    the column at that index
    - column_headers: A list of the names of the features in order
    - max_depth=1: The maximum depth of the tree
    - method='ENT': 'ENT'->Entropy, 'ME'->Majority Error, or 'GINI'->Gini index
    - unknown_vals=False: Whether or not to replace instances of 'unknown' in the data set
    with the most common label
    - numerical_vals=False: Whether or not to replace numerical attributes with a boolean value indicating
    that that value is less than or greater than the mean

2. Classify Data
    Call the classify_data function of the DecisionTree object created in the previous step
    Parameters:
    - row: the row of data in the same order as the training set

How to use EnsembleLearning:
----------------------------------
- Each of the ada_boost.py (AdaBoost class), bagging.py (Bagging class), and random_forest.py (RandomForest) runs the algorithm corresponding to the name of the file.
- Each of the classes has a classify_data function which can be used after the training has completed.
    - Parameters:
        - row: the row of data in the same order as the training set
        - label_map (AdaBoost): a dictionary mapping with 'positive' and 'negative' as the keys and the label that corresponds to the positive or negative values as the values.

AdaBoost Parameters:
----------------------------------
- data: the training data
- possible_attribute_values: A dictionary with the column headers as the keys and a list of possible values for that column as the value
- column_headers: a list of the column headers
- T: the number of weak classifiers to be used

Bagging Parameters:
----------------------------------
- data: the training data
- possible_attribute_values: A dictionary with the column headers as the keys and a list of possible values for that column as the value
- column_headers: a list of the column headers
- num_weak_classifiers: the number of weak classifiers to be used
- num_samples: the number of random rows to use from the training data for each iteration

RandomForest Parameters:
----------------------------------
- data: the training data
- possible_attribute_values: A dictionary with the column headers as the keys and a list of possible values for that column as the value
- column_headers: a list of the column headers
- num_weak_classifiers: the number of weak classifiers to be used
- G: the number of random rows to use from the training data for each iteration


How to use LinearRegresion:
----------------------------------
- The class LinearRegression creates a classifier for either batch or or stochastic gradient descent
Parameters:
    - data: the training data
    - type: either 'BATCH' or 'STOCHASTIC'
- It has a classify_data function which can be used after the training has completed.
    - Parameters:
        - row: the row of data in the same order as the training set

How to use Perceptron:
----------------------------------
- The class Perceptron creates a classifier using perceptron. It supports functionality for the standard, voted, and average versions
- The __init__ function will automatically trigger the training of the model
Parameters:
    - data: the training data
    - T: the number of epochs
    - type: either 'STANDARD', 'VOTED', OR 'AVERAGE'
- It has a classify_data function which can be used after the training has completed.
    - Parameters:
        - row: the row of data in the same order as the training set

How to use SVM:
----------------------------------
- The class SVM implements various forms of SVM. It supports functionality for the primal form, dual form, and dual form with a kernal
- The __init__ function will automatically trigger the training of the model
Parameters:
T, C, a, gamma_start, type='PRIMAL', learning_schedule=1
    - data: the training data
    - T: the number of epochs
    - C: the hyper parameter C
    - a: the hyper parameter a
    - gamma_start: the starting value for gamma (updated each epoch)
    - type: either 'PRIMAL', or 'DUAL'
    - learning_schedule: the two types of learning schedules requested in the assignment
- It has a classify_data function which can be used after the training has completed.
    - Parameters:
        - row: the row of data in the same order as the training set

How to use NeuralNetworks:
----------------------------------
- The class NeuralNet implements a 3-layer neural network. It uses the sigmoid function as the activation function
- The __init__ function will automatically trigger the training of the model
Parameters:
data, T, gamma_start=.1, d=.1, hidden_neurons=None
    - data: the training data
    - T: the number of epochs
    - gamma_start: the starting value for gamma (updated each epoch)
    - d: hyper parameter d, used to update gamma
    - hidden_neurons: the number of neurons to use in a hidden layer (default is the same as the number of input features plus a bias node)
- It has a classify_data function which can be used after the training has completed.
    - Parameters:
        - row: the row of data in the same order as the training set
