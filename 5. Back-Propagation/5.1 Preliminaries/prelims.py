"""
    EXTERNAL LIBRARIES
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math


"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import MultiLayerNetwork
from utils.auxfunctions import sigmoid


"""
    In the forward pass, the network computes an output based on its
    current inputs. Each node, i, computes a weighted sum Ai of its
    inputs and passes this through a nonlinearity to obtain the node
    output Yi.
"""

# Create a subclass that adds a feedforward function to the
# basic mult-layer network class setup in the auxfunctions module
class FeedforwardFunctionNetwork(MultiLayerNetwork):
    
    def feedforward(self, inputs):
        # Fill the input nodes
        if self.bias:
            for input_ in inputs:
                input_.append(1.0)
        self.inputNodes = np.asarray(inputs)
        # Fill in the hidden nodes
        prevLayer = self.inputNodes
        for i in range(len(self.hiddenNodes)):
            self.hiddenNodes[i] = np.dot(
                prevLayer, self.weights[i]
            )
            prevLayer = activationFunction(self.hiddenNodes[i])
        # Fill in the output nodes
        self.outputNodes = np.dot(
            prevLayer, self.weights[-1]
        )
        # Return the activated output layer
        return activationFunction(self.outputNodes)

# Construct the network
network = FeedforwardFunctionNetwork(
    2, [3, 3], 1, sigmoid, bias=True, randomize=True
)

# Perform the feedforward function on a given set of inputs
networkOutput = network.feedforward(
    [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ]
)

# Print out the network's output
print('\n\nThe network\'s output on the given input is:\n')
print(networkOutput)
print('\n')


"""
    Unless the network is perfectly trained, the network outputs will
    differ somewhat from the desired outputs. The significance of these
    differences is measured by an error (or cost) function, E. In the
    following code, we use the SSE error function.
"""

# Create a subclass that adds an SSE error function to the multi-layer
# network subclass created above that already contains a
# feedforward function.
class SSEFunctionNetwork(FeedforwardFunctionNetwork):

    def SSE(self, inputs, targetOutputs, activationFunction):
        pass