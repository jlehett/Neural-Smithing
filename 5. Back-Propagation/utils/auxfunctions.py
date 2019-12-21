"""
    Imported libraries
"""

import numpy as np
from math import e


"""
    Aux Functions
"""

# Define sigmoid activation function
@np.vectorize
def sigmoid(x):
    return 1.0 / (1.0 + e ** (-x))

# Define sigmoid derivative function
@np.vectorize
def sigmoidDerivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


"""
    Multi-Layer Network class 
"""

class MultiLayerNetwork:

    def __init__(self, inputSize, hiddenSizes, outputSize,
                 activationFunction, activationDerivative,
                 bias=True, randomize=True):
        # Set class activation properties
        self.activationFunction = activationFunction
        self.activationDerivative = activationDerivative
        # If bias is set to true, we must add 1 to the input size
        if bias:
            inputSize += 1
        self.bias = bias
        # Create nodes of the network
        self.createNodes(inputSize, hiddenSizes, outputSize)
        # Create weights for the network
        self.createWeights(inputSize, hiddenSizes, outputSize, randomize)

    def createNodes(self, inputSize, hiddenSizes, outputSize):
        # Create input nodes
        self.inputNodes = np.zeros((inputSize))
        # Create hidden nodes
        self.hiddenNodes = []
        for hiddenSize in hiddenSizes:
            self.hiddenNodes.append(
                np.zeros((hiddenSize))
            )
        # Create output nodes
        self.outputNodes = np.zeros((outputSize))

    def createWeights(self, inputSize, hiddenSizes, outputSize, randomize):
        # Put all of the sizes into a single list for easy access
        allSizes = hiddenSizes
        allSizes.insert(0, inputSize)
        allSizes.append(outputSize)
        # Create all weight layers
        self.weights = []
        for i in range(1, len(allSizes)):
            prevLayerSize = allSizes[i-1]
            currentLayerSize = allSizes[i]
            self.weights.append(
                self.createWeightLayer(
                    (prevLayerSize, currentLayerSize), 
                    randomize
                )
            )

    def createWeightLayer(self, size, randomize):
        # Create a single weight layer given the size and whether the
        # weights should be randomized
        if randomize:
            return np.random.normal(size=(size[0], size[1]), scale=0.3)
        else:
            return np.zeros(size)