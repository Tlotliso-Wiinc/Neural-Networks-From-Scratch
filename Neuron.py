import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        # Convert weights to numpy array if not already
        self.weights = np.array(weights)
        self.bias = bias

    def feedforward(self, inputs):
        # Convert inputs to numpy array if not already
        inputs = np.array(inputs)
        # Use numpy dot product for efficient computation
        return np.dot(inputs, self.weights) + self.bias
