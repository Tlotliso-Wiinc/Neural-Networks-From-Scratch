import numpy as np

class ReLUActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output
