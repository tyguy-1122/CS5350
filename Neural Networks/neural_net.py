import random
import math

def activation_func(z):
    return 1 / (1 + math.exp(-z))

def activation_func_prime(upper_node, lower_node):
    return upper_node * (1 - upper_node) * lower_node

class NeuralNet():
    def __init__(self, data, hidden_neurons=None):
        self.data = data
        if hidden_neurons == None:
            hidden_neurons = len(data[0] - 1)
        self.create_classifier(data)
        self.data = [[1, 1, 1]] # only here until further along in the implementation
        self.hidden_neurons = 2 # only here until further along in the implementation

    def create_classifier(self, data):
        pass

    def forward_propagate(self):
        self.input_nodes = [1, 1, 1]
        # Layer 1
        self.layer_1_nodes = [0 for i in range(self.hidden_neurons + 1)]
        self.layer_1_nodes[0] = 1
        for i in range(self.hidden_neurons):
            node_val = 0
            for j in range(len(self.input_nodes)):
                node_val += self.layer_1_weights[j][i] * self.input_nodes[j]
            self.layer_1_nodes[i+1] = activation_func(node_val)

        # Layer 2
        self.layer_2_nodes = [0 for i in range(self.hidden_neurons + 1)]
        self.layer_2_nodes[0] = 1
        for i in range(self.hidden_neurons):
            node_val = 0
            for j in range(self.hidden_neurons + 1):
                node_val += self.layer_2_weights[j][i] * self.layer_1_nodes[j]
            self.layer_2_nodes[i+1] = activation_func(node_val)

        # Layer 3 (output)
        node_val = 0
        for j in range(self.hidden_neurons + 1):
            node_val += self.layer_3_weights[j][0] * self.layer_2_nodes[j]
        self.output_node = node_val

        # self.layer_1_nodes = [1, .00247, .9975]
        # self.layer_2_nodes = [1, .018, .982]
        # self.output_node = -2.437

    def back_propagate(self, y_true):
        self.layer_1_gradients = [[0 for i in range(self.hidden_neurons)] for j in range(len(self.data[0]))]
        self.layer_2_gradients = [[0 for i in range(self.hidden_neurons)] for j in range(self.hidden_neurons + 1)]
        self.layer_3_gradients = [[0] for j in range(self.hidden_neurons + 1)]
        self.layer_3_cache = 0
        self.layer_2_cache = [0 for i in range(len(self.layer_2_nodes)-1)]

        for i in range(len(self.layer_3_gradients)):
            for j in range(len(self.layer_3_gradients[i])):
                grad = (self.output_node - y_true)
                self.layer_3_cache = grad
                self.layer_3_gradients[i][j] = grad * self.layer_2_nodes[i]

        for i in range(len(self.layer_2_gradients)):
            for j in range(len(self.layer_2_gradients[i])):
                grad = self.layer_3_cache * self.layer_3_weights[j+1][0]
                self.layer_2_cache[j] = grad
                self.layer_2_gradients[i][j] = grad * activation_func_prime(self.layer_2_nodes[j+1], self.layer_1_nodes[i])

        for i in range(len(self.layer_1_gradients)):
            for j in range(len(self.layer_1_gradients[i])):
                grad = 0
                for k in range(self.hidden_neurons):
                    grad += self.layer_2_cache[k] * self.layer_2_weights[j+1][k] * activation_func_prime(self.layer_1_nodes[j+1], self.input_nodes[i])
                self.layer_1_gradients[i][j] = grad
        
    def init_weights(self):
        self.layer_1_weights = [[random.random() for i in range(self.hidden_neurons)] for j in range(len(self.data))]
        self.layer_2_weights = [[random.random() for i in range(self.hidden_neurons)] for j in range(self.hidden_neurons + 1)]
        self.layer_3_weights = [[random.random()] for j in range(self.hidden_neurons + 1)]

        # Remove this later!!
        self.layer_1_weights = [[-1, 1],[-2, 2],[-3, 3]]
        self.layer_2_weights = [[-1, 1],[-2, 2],[-3, 3]]
        self.layer_3_weights = [[-1],[2],[-1.5]]

if __name__ == '__main__':
    nn = NeuralNet(None, hidden_neurons=2)
    nn.init_weights()
    nn.forward_propagate()
    nn.back_propagate(1)
