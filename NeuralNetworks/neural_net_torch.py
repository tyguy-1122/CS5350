import torch
from torch import nn, optim

class NeuralNetTorch(nn.Module):
    def __init__(self, ):
        super(NeuralNetTorch, self).__init__()

        # Define the layers
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3

        # Init the weights
        self.W1 = torch.randn(self.input_size, self.hidden_size)
        self.W2 = torch.randn(self.hidden_size, self.output_size)
