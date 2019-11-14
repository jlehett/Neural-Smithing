"""
    EXTERNAL LIBRARIES
"""

import numpy as np
import random
from math import sqrt

"""
    Single-Layer Network Class
"""
class SingleLayerNetwork:

    def __init__(self, numInputs, numOutputs, activationFunc,
                randomize=False, bias=False):
        """
            Construct a single-layer neural network given the
            number of input nodes, the number of output nodes,
            and the activation function to be used in the output
            layer.

            randomize:  If set to true, the starting weights are
                        randomized in the range [-1, +1]
        """
        # Parameters
        self.activationFunc = activationFunc
        self.bias = bias
        # Set weights according to randomize parameter
        if bias:
            self.weights = np.zeros((numOutputs, numInputs+1))
        else:
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
        if self.bias:
            inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        self.outputNodes = np.dot(inputs, self.weights.T)
        for i in range(len(self.outputNodes)):
            self.outputNodes[i] = self.activationFunc(self.outputNodes[i])
        return self.outputNodes

    def getWeights(self):
        """
            Return the networks current weights in a numpy array of size
            ( self.outputNodes.shape[0], self.inputNodes.shape[0] )
        """
        return self.weights

    def getWeightMagnitude(self, inputNum=0):
        """
            Return the networks current weight magnitude for the specified
            input (or the first input if no argument is given)
        """
        denominator = 0.0
        for w in self.weights[inputNum]:
            denominator += w ** 2.0
        return sqrt(denominator)


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