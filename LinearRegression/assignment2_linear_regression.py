from DecisionTree.utilities import extract_data_from_csv
from LinearRegression.linear_regression import LinearRegression, calc_vector_inner_prod, compute_cost_function
import matplotlib.pyplot as plt
import numpy as np

############################################
############################################
# Problem 4 - Linear Regression
############################################
############################################

############################################
# Problem 4 - A: Batch Gradient Descent
############################################

# Read data from .csv for each iteration
slump_data_training_str = extract_data_from_csv('./slump/train.csv')
slump_data_testing_str = extract_data_from_csv('./slump/test.csv')

slump_data_training = []
for row in slump_data_training_str:
    float_row = []
    for x in row:
        float_row.append(float(x))
    slump_data_training.append(float_row)

slump_data_testing = []
for row in slump_data_testing_str:
    float_row = []
    for x in row:
        float_row.append(float(x))
    slump_data_testing.append(float_row)

# Get the correct labels for the testing data for calculating classification error
true_labels = [row[-1] for row in slump_data_testing]
# Build the classifier with the specified number of weak classifiers
classifier = LinearRegression(slump_data_training)

# Get the classification error
plt.plot(classifier.cost_function_values)
plt.ylabel('Cost Function Value')
plt.xlabel('Step Number')
plt.title('Batch Gradient Descent Cost Function Progression')
plt.show()
classification_error = compute_cost_function(classifier.classifier, slump_data_testing)
print('BATCH GRADIENT DESCENT')
print('---------------------------')
print('Cost function value on testing data - ', classification_error)
print('R value at start - .1')
print('R value multiplier per step - .9999')
print(f'Final classifier: {classifier.classifier}')

#############################################
# Problem 4 - A: Stochastic Gradient Descent
#############################################

# Read data from .csv for each iteration
slump_data_training_str = extract_data_from_csv('./slump/train.csv')
slump_data_testing_str = extract_data_from_csv('./slump/test.csv')

slump_data_training = []
for row in slump_data_training_str:
    float_row = []
    for x in row:
        float_row.append(float(x))
    slump_data_training.append(float_row)

slump_data_testing = []
for row in slump_data_testing_str:
    float_row = []
    for x in row:
        float_row.append(float(x))
    slump_data_testing.append(float_row)

# Get the correct labels for the testing data for calculating classification error
true_labels = [row[-1] for row in slump_data_testing]
# Build the classifier with the specified number of weak classifiers
classifier = LinearRegression(slump_data_training, type='STOCHASTIC')

# Get the classification error
plt.plot(classifier.cost_function_values)
plt.ylabel('Cost Function Value')
plt.xlabel('Step Number')
plt.title('Stochastic Gradient Descent Cost Function Progression')
plt.show()
classification_error = compute_cost_function(classifier.classifier, slump_data_testing)
print('STOCHASTIC GRADIENT DESCENT')
print('---------------------------')
print('Cost function value on testing data- ', classification_error)
print('R value at start - .1')
print('R value multiplier per step - .9999')
print(f'Final classifier: {classifier.classifier}')


#######################################
# Analytical Solution to weight vector
#######################################

slump_data_training_str = extract_data_from_csv('./slump/train.csv')
slump_data_training = []
for row in slump_data_training_str:
    float_row = []
    for x in row:
        float_row.append(float(x))
    slump_data_training.append(float_row)

X = np.transpose(np.array([row[:-1] for row in slump_data_training]))
Y = np.array([row[-1] for row in slump_data_training])

w_analytical = np.dot(np.linalg.inv(np.dot(X, np.transpose(X))), np.dot(X, Y))

print(f'The analytical solution to the optimal weight vector is: \n{w_analytical}')

