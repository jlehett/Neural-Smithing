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
from utils.auxfunctions import FeedforwardNetwork
from utils.auxfunctions import sigmoid


"""
    In the forward pass, the network computes an output based on its current inputs.
    Each node, i, computes a weighted sum Ai of its inputs an passes this through a
    nonlinearity to obtain the node output Yi.

    This process is detailed below. The FeedforwardNetwork class only contains an init
    function that stores the weights of each layer in an array. The code written below
    performs a forward pass on the network.
"""

# Feedforward function for determining output of network given an input
def feedforward(inputs, network, activationFunction):
    # convert inputs to numpy array for easier calculations
    inputs = np.asarray(inputs)
    # Iterate through each layer of the network
    currentLayer = inputs
    for layerNum in range(len(network.weights)):
        # Perform weighted addition
        currentLayer = np.dot(currentLayer, network.weights[layerNum])
        # Apply activation function
        currentLayer = activationFunction(currentLayer)
    # Return the final output
    return currentLayer

# Define a network with an input size of 5, 3 hidden layers each with 5
# nodes, and an output layer of size 3.
ffn = FeedforwardNetwork(
    5, [5, 5, 5], 3, randomize=True
)

# Print out the resulting output of the network
output = feedforward([1, 1, 1, 1, 1], ffn, sigmoid)

print('\nThe resulting output for the neural network with input vector ' +
      '[1, 1, 1, 1, 1] is: ' + str(output) + '\n\n')


"""
    Unless the network is perfectly trained, the network outputs will
    differ somewhat from the desired outputs. The significance of these
    differences is measured by an error (or cost) function E. In this
    section, we will use the SSE error function.
"""

# Define the SSE error function
def SSE(inputs, network, targetOutputs):
    # Convert outputs to numpy array for easier calculations
    targetOutputs = np.asarray(targetOutputs)
    # First, compute the network outputs on the given inputs
    networkOutputs = feedforward(inputs, network, sigmoid)
    # Compute the SSE for the network on the given inputs
    
    # Compute the squared difference of each output to its corresponding
    # targets. This is done easily using numpy arrays.
    squaredDifferenceArray = (networkOutputs - targetOutputs) ** 2
    # Add each of these errors up. Again this is done easily in numpy.
    totalError = np.sum(squaredDifferenceArray)
    # Divide the total error by 2 and you have the final resulting SSE.
    totalError /= 2.0
    return totalError

# Test SSE
error = SSE(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 0, 1]
            ],
            ffn,
            [
                [1, 0, 1],
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 1]
            ]
        )

print('The resulting error for the neural network provided in section 2 ' +
      'of the code is: ' + str(error) + '\n\n')