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
from utils.auxfunctions import sigmoid, sigmoidDerivative


"""
    In the forward pass, the network computes an output based on its
    current inputs. Each node, i, computes a weighted sum Ai of its
    inputs and passes this through a nonlinearity to obtain the node
    output Yi.
"""

# Create a subclass that adds a feedforward function to the
# basic multi-layer network class setup in the auxfunctions module
class FeedforwardFunctionNetwork(MultiLayerNetwork):
    
    def feedforward(self, inputs):
        new_inputs = []
        # Fill the input nodes
        for input_ in inputs:
            new_inputs.append([])
            for value in input_:
                new_inputs[-1].append(value)
            if self.bias:
                new_inputs[-1].append(1.0)
        self.inputNodes = np.asarray(new_inputs)
        # Fill in the hidden nodes
        prevLayer = self.inputNodes
        for i in range(len(self.hiddenNodes)):
            self.hiddenNodes[i] = np.dot(
                prevLayer, self.weights[i]
            )
            prevLayer = self.activationFunction(self.hiddenNodes[i])
        # Fill in the output nodes
        self.outputNodes = np.dot(
            prevLayer, self.weights[-1]
        )
        # Return the activated output layer
        return self.activationFunction(self.outputNodes)

if __name__ == '__main__':
    # Construct the network
    network = FeedforwardFunctionNetwork(
        2, [3, 3], 1, sigmoid, sigmoidDerivative, bias=True, randomize=True
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

    def SSE(self, inputs, targetOutputs):
        # Convert target outputs to numpy array
        targetOutputs = np.asarray(targetOutputs)
        # Get the network outputs through feedforward function
        networkOutputs = self.feedforward(inputs)
        # Obtain the error function specified
        sse = np.sum(
            (targetOutputs - networkOutputs) ** 2
        )
        # Return the final sse value
        return sse

if __name__ == '__main__':
    # Construct the network
    network = SSEFunctionNetwork(
        2, [3, 3], 1, sigmoid, sigmoidDerivative, bias=True, randomize=True
    )

    # Perform the SSE function on a given set of inputs
    networkSSE = network.SSE(
        [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ],
        [
            [0],
            [0],
            [0],
            [1]
        ]
    )

    # Print out the network's SSE
    print('\n\nThe network\'s SSE on the given input is:\n')
    print(networkSSE)
    print('\n')