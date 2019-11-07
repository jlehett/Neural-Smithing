"""
    EXTERNAL LIBRARIES
"""

import numpy as np
import random

"""
    Single-Layer Network Class
"""
class SingleLayerNetwork:

    def __init__(self, numInputs, numOutputs, activationFunc,
                randomize=False):
        """
            Construct a single-layer neural network given the
            number of input nodes, the number of output nodes,
            and the activation function to be used in the output
            layer.

            randomize:  If set to true, the starting weights are
                        randomized in the range [-1, +1]
        """
        # Parameters
        self.outputNodes = np.zeros((numOutputs))
        self.activationFunc = activationFunc
        # Set weights according to randomize parameter
        self.weights = np.zeros((numOutputs, numInputs))
        if randomize:
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    self.weights[i, j] = random.random() * 2.0 - 1.0

    def evaluate(self, inputs):
        """
            Evaluate the inputs passing through the neural network
            using the network's current weights.
        """
        for i in range(self.outputNodes.shape[0]):
            self.outputNodes[i] = 0
            for j in range(inputs.shape[0]):
                self.outputNodes[i] += inputs[j] * self.weights[i, j]
            self.outputNodes[i] = self.activationFunc(self.outputNodes[i])
        return self.outputNodes

    def getWeights(self):
        """
            Return the networks currents weights in a numpy array of size
            ( self.outputNodes.shape[0], self.inputNodes.shape[0] )
        """
        return self.weights


"""
    TESTING
"""
from math import e

if __name__ == '__main__':

    # Sigmoid activation function
    def sigmoid(x):
        return 1.0 / (1.0 + e ** (-x))

    sln = SingleLayerNetwork(3, 5, sigmoid, randomize=True)

    print(sln.evaluate(
        np.array([5, 1, 2])
        ))