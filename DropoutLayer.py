import numpy as np

class DropoutLayer:
    def __init__(self, rate):
        self.rate = 1 - rate
        self.mask = None

    # Forward pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save the scaled binary mask
        self.mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply the mask to output values
        self.output = inputs * self.mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.mask