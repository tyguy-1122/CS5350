import random
import copy
import numpy as np
import math
from scipy.optimize import minimize

def calc_learning_rate_one(y_start, t, a):
    return y_start / (1 + ((y_start / a) * t))

def calc_learning_rate_two(y_start, t, a):
    return y_start / (1 + t)

def dot_prod(weight_vector, row):
    dot_prod = 0
    for i in range(len(weight_vector)):
        dot_prod += weight_vector[i] * row[i]
    
    return dot_prod

def gaussian_kernal(x_i, x_j, gamma):
    return math.exp(-1 * (np.linalg.norm(np.subtract(x_i, x_j)) ** 2) / gamma)

def dual_objective_function(a, X, Y):
    value = 0
    for i in range(len(X)):
        for j in range(len(X)):
            value += Y[i] * Y[j] * a[i] * a[j] * np.dot(X[i].T, X[j])
    return .5 * value - a.sum()

def incorrect_objective_function(val, a, X, Y):
    return .5 * np.sum(np.tile(np.array([Y]).transpose(), len(Y))@Y) * np.sum(np.tile(np.array([a]).transpose(), len(a))@a) * np.sum(X.T@X) - a.sum()

def gaussian_objective_function(a, X, Y, gamma):
    value = 0
    for i in range(len(X)):
        for j in range(len(X)):
            value += Y[i] * Y[j] * a[i] * a[j] * gaussian_kernal(X[i], X[j], gamma)
    return .5 * value - a.sum()

def constrain_sum_a_y(a, Y):
    return np.dot(a, Y)

def constrain_a_less_c(a, C):
    return C - a

def constrain_a_greater_zero(a):
    return a

class SVM:
    def __init__(self, data, T, C, a, gamma_start, type='PRIMAL', learning_schedule=1):
        self.type = type
        self.C = C
        self.gamma_start = gamma_start
        self.a = a

        if (learning_schedule == 1):
            self.learning_schedule_function = calc_learning_rate_one
        elif (learning_schedule == 2):
            self.learning_schedule_function = calc_learning_rate_two
        else:
            raise Exception('Invalid learning schedule parameter')

        if type == 'PRIMAL':
            self.create_primal_classifier(data, T)
        elif type == 'DUAL' or type == 'GAUSSIAN':
            self.create_dual_classifier(data)
        else:
            raise Exception('Invalid type')

    def create_primal_classifier(self, data, T):
        self.classifier_history = []
        self.classifier = [0] * (len(data[0]) - 1)

        for t in range(T):
            self.classifier_history.append(self.classifier)
            random.shuffle(data)
            learning_rate = self.learning_schedule_function(self.gamma_start, t, self.a)

            num_updates = 0
            # Look at each example in training data
            for j in range(len(data)):
                row = data[j][:-1]
                prediction = dot_prod(self.classifier, row)
                true_label = data[j][-1]

                # Data not on or outside margin
                if true_label * prediction <= 1:
                    num_updates +=1
                    first_term = [learning_rate * x for x in self.classifier]
                    second_term = [true_label * self.C * len(data) * learning_rate * x for x in row]
                    for k in range(len(self.classifier)):
                        self.classifier[k] += -1 * first_term[k] + second_term[k]
                # Data outside margin
                else:
                    for k in range(len(self.classifier) - 1):
                        self.classifier[k+1] = self.classifier[k+1] * (1 - learning_rate)
            
    def create_dual_classifier(self, data):
        X = np.array([row[1:-1] for row in data])[:100]
        Y = np.array([row[-1] for row in data])[:100]
        a = np.random.rand(100)
        constraints = (
            {'type': 'eq', 'fun': constrain_sum_a_y, 'args': [Y]},
            {'type': 'ineq', 'fun': constrain_a_greater_zero},
            {'type': 'ineq', 'fun': constrain_a_less_c, 'args': [self.C]})

        if self.type == 'DUAL':
            result = minimize(dual_objective_function, a, args=(X, Y), method='SLSQP', constraints=constraints)
        else:
            result = minimize(gaussian_objective_function, a, args=(X, Y, self.gamma_start), method='SLSQP', constraints=constraints)
        alphas = result.x
        alphas[np.isclose(alphas, 0)] = 0

        # Calculate the weight vector
        self.classifier = [0] * (len(X[0])) # Not worried about bias yet

        for i in range(100):
            for j in range(len(self.classifier)):
                self.classifier[j] += alphas[i] * Y[i] * X[i][j]
        
        # Calculate bias and append
        for i in range(100):
            if alphas[i] > 0:
                b = Y[i] - np.dot(np.array(self.classifier), X[i])
                self.classifier.insert(0, b)
                break

    def classify_data(self, row):
        prod = dot_prod(self.classifier, row)
        
        return -1 if prod < 0 else 1
    