from math import sqrt
from random import randrange

def compute_cost_function(W, data):
    error = 0
    for row in data:
        error += .5 * (row[-1] - calc_vector_inner_prod(W, row)) ** 2
    return error

def calc_vector_inner_prod(vec1, vec2):
    product = 0
    for i in range(len(vec1)):
        product += vec1[i] * vec2[i]
    return product

def calc_vector_norm(vec):
    sum_squares = 0
    for component in vec:
        sum_squares += component ** 2
    return sqrt(sum_squares)

class LinearRegression:
    def __init__(self, data, type='BATCH'):
        self.cost_function_values = []
        if type == 'BATCH':
            self.create_classifier_batch(data)
        elif type == 'STOCHASTIC':
            self.create_classifier_stochastic(data)
        else:
            raise ValueError('Invalid linear regression type! Must be "BATCH" or "STOCHASTIC".')
    
    def create_classifier_batch(self, data):
        threshold = 1e-6
        R = .1
        # Initialize W
        W = []
        for i in range(len(data[0]) - 1):
            W.append(0)
        # Initialize the norm difference (||W_t - W_{t-1}||) to some value greater than threshold
        norm_difference = 1

        # Get initial cost_function value
        self.cost_function_values.append(compute_cost_function(W, data))

        # Begin Gradient Descent
        while norm_difference > threshold:
            # Calculate the gradient
            gradient_vec = []
            for i in range(len(W)):
                sum = 0
                for j in range(len(data)):
                    sum -= (data[j][-1] - calc_vector_inner_prod(W, data[j])) * data[j][i]
                gradient_vec.append(sum/len(data))
            
            # Create the next weight vector
            next_W = []
            for i in range(len(W)):
                next_W.append(W[i] - R * gradient_vec[i])
            
            # Calculate the new norm difference
            difference = []
            for i in range(len(W)):
                difference.append(next_W[i] - W[i])
            norm_difference = calc_vector_norm(difference)

            W = next_W

            self.cost_function_values.append(compute_cost_function(W, data))
            R *= .9999
        self.classifier = W
    
    def create_classifier_stochastic(self, data):
        threshold = 1e-6
        R = .1

        # Initialize W
        W = []
        for i in range(len(data[0]) - 1):
            W.append(0)
        # Initialize the norm difference (||W_t - W_{t-1}||) to some value greater than threshold
        norm_difference = 1

        # Get initial cost_function value
        self.cost_function_values.append(compute_cost_function(W, data))

        # Begin Gradient Descent
        while norm_difference > threshold:
            # Calculate the gradient
            gradient_vec = []
            for i in range(len(W)):
                sum = 0
                rand_row = randrange(0, len(data))
                sum -= (data[rand_row][-1] - calc_vector_inner_prod(W, data[rand_row])) * data[rand_row][i]
                gradient_vec.append(sum)
            
            # Create the next weight vector
            next_W = []
            for i in range(len(W)):
                next_W.append(W[i] - R * gradient_vec[i])
            
            # Calculate the new norm difference
            difference = []
            for i in range(len(W)):
                difference.append(next_W[i] - W[i])
            norm_difference = calc_vector_norm(difference)

            W = next_W

            self.cost_function_values.append(compute_cost_function(W, data))
            R *= .9999
        self.classifier = W

    def classify_data(self, row):
        return calc_vector_inner_prod(self.classifier, row)


