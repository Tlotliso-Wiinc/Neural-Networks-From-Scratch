class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def feedforward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feedforward(inputs))
        return outputs