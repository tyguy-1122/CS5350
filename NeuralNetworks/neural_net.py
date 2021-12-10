import random
import numpy as np

def activation_func(z):
    return 1 / (1 + np.exp(-z))

def activation_func_prime(upper_node, lower_node):
    return upper_node * (1 - upper_node) * lower_node

class NeuralNet():
    def __init__(self, data, T, gamma_start=.1, d=.1, hidden_neurons=None):
        self.data = data
        self.gamma_start = gamma_start
        self.d = d
        self.T = T
        if hidden_neurons == None:
            self.hidden_neurons = len(data[0]) - 1
        else:
            self.hidden_neurons = hidden_neurons
        self.create_classifier()

    def create_classifier(self):
        # Init parameters
        self.init_weights()

        # Learn the training data
        for t in range(self.T):
            random.shuffle(self.data)
            gamma = self.gamma_start / (1 + (self.gamma_start / self.d) * t)
            for i in range(len(self.data)):
                self.forward_pass(self.data[i])
                self.back_propagate(self.data[i][-1])
                self.update_weights(gamma)

    def forward_pass(self, row):
        self.input_nodes = row[:-1]

        # Layer 1
        self.layer_1_nodes = [0 for i in range(self.hidden_neurons)]
        self.layer_1_nodes[0] = 1
        for i in range(self.hidden_neurons - 1):
            node_val = 0
            for j in range(len(self.input_nodes)):
                node_val += self.layer_1_weights[j][i] * self.input_nodes[j]
            self.layer_1_nodes[i+1] = activation_func(node_val)

        # Layer 2
        self.layer_2_nodes = [0 for i in range(self.hidden_neurons)]
        self.layer_2_nodes[0] = 1
        for i in range(self.hidden_neurons - 1):
            node_val = 0
            for j in range(self.hidden_neurons):
                node_val += self.layer_2_weights[j][i] * self.layer_1_nodes[j]
            self.layer_2_nodes[i+1] = activation_func(node_val)

        # Layer 3 (output)
        node_val = 0
        for j in range(self.hidden_neurons):
            node_val += self.layer_3_weights[j][0] * self.layer_2_nodes[j]
        self.output_node = node_val

    def update_weights(self, gamma):
        # Layer 1
        for i in range(len(self.layer_1_weights)):
            for j in range(len(self.layer_1_weights[i])):
                self.layer_1_weights[i][j] -= gamma * self.layer_1_gradients[i][j]

        # Layer 2
        for i in range(len(self.layer_2_weights)):
            for j in range(len(self.layer_2_weights[i])):
                self.layer_2_weights[i][j] -= gamma * self.layer_2_gradients[i][j]
            
        # Layer 3
        for i in range(len(self.layer_3_weights)):
            for j in range(len(self.layer_3_weights[i])):
                self.layer_3_weights[i][j] -= gamma * self.layer_3_gradients[i][j]
            
    def back_propagate(self, y_true):
        self.layer_1_gradients = [[0 for i in range(self.hidden_neurons-1)] for j in range(len(self.input_nodes))]
        self.layer_2_gradients = [[0 for i in range(self.hidden_neurons-1)] for j in range(self.hidden_neurons)]
        self.layer_3_gradients = [[0] for j in range(self.hidden_neurons)]
        self.layer_3_cache = 0
        self.layer_2_cache = [0 for i in range(len(self.layer_2_nodes)-1)]

        # Layer 3
        for i in range(len(self.layer_3_gradients)):
            for j in range(len(self.layer_3_gradients[i])):
                grad = (self.output_node - y_true)
                self.layer_3_cache = grad
                self.layer_3_gradients[i][j] = grad * self.layer_2_nodes[i]

        # Layer 2
        for i in range(len(self.layer_2_gradients)):
            for j in range(len(self.layer_2_gradients[i])):
                grad = self.layer_3_cache * self.layer_3_weights[j+1][0]
                self.layer_2_cache[j] = grad
                self.layer_2_gradients[i][j] = grad * activation_func_prime(self.layer_2_nodes[j+1], self.layer_1_nodes[i])

        # Layer 1
        for i in range(len(self.layer_1_gradients)):
            for j in range(len(self.layer_1_gradients[i])):
                grad = 0
                for k in range(self.hidden_neurons - 1):
                    grad += self.layer_2_cache[k] * self.layer_2_weights[j+1][k] * activation_func_prime(self.layer_1_nodes[j+1], self.input_nodes[i])
                self.layer_1_gradients[i][j] = grad
        
    def init_weights(self):
        self.layer_1_weights = [[np.random.normal() for i in range(self.hidden_neurons-1)] for j in range(len(self.data[0]) - 1)]
        self.layer_2_weights = [[np.random.normal() for i in range(self.hidden_neurons-1)] for j in range(self.hidden_neurons)]
        self.layer_3_weights = [[np.random.normal()] for j in range(self.hidden_neurons)]

        # self.layer_1_weights = [[0 for i in range(self.hidden_neurons-1)] for j in range(len(self.data[0]) - 1)]
        # self.layer_2_weights = [[0 for i in range(self.hidden_neurons-1)] for j in range(self.hidden_neurons)]
        # self.layer_3_weights = [[0] for j in range(self.hidden_neurons)]

    def classify_data(self, row):
        self.forward_pass(row)
        return 1 if self.output_node >= 0 else -1

