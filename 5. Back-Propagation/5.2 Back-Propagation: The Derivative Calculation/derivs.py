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
from utils.auxfunctions import sigmoid, sigmoidDerivative
from utils.auxfunctions import feedforward, SSE


"""
    Having obtained the outputs and calculated the error, the next step
    is to calculate the derivative of the error with respect to the
    weights. The total derivative is just the sum of the per-pattern
    derivatives.

    For output nodes, the delta value depends only on the 
"""

# Define a back-propagation function using SSE as the error function
def backpropagate(
    network, inputs, targetOutputs,
    activationFunction, activationDerivative
    ):
    # Convert target outputs to numpy array
    targetOutputs = np.asarray(targetOutputs)
    # First, obtain the outputs of the network on the given inputs
    networkOutputs, layerOutputs = feedforward(inputs, network, activationFunction)
    # Get the SSE for the network on the given inputs
    sse = SSE(inputs, network, targetOutputs, sigmoid)
    # Get the delta value for the output nodes
    deltas = []
    deltas.append(
        -(targetOutputs - networkOutputs) * activationDerivative(networkOutputs)
    )
    deltas[0] = np.sum(deltas[0], axis=0)
    # Obtain the delta values for all of the hidden layers
    layerIndex = -2
    for i in range(len(network.weights) - 1):
        deltas.append(
            np.sum(activationDerivative(layerOutputs[layerIndex]), axis=0) *
            np.dot(network.weights[layerIndex+1], deltas[-1])
        )
        layerIndex -= 1
    # Reverse the deltas list so it is facing the correct direction.
    deltas.reverse()
    # Multiply each delta by the corresponding node outputs to obtain the
    # final derivative calculation
    for deltaIndex in range(len(deltas)):
        deltas[deltaIndex] *= np.sum(layerOutputs[deltaIndex], axis=0)
    # Return the final derivative deltas
    return deltas

# First, define a network
network = FeedforwardNetwork(
    3, [4, 5], 3, randomize=True
)

# Define a set of inputs
inputs = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]

# Define a set of target outputs
targetOutputs = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
]

deltas = backpropagate(
    network, inputs, targetOutputs, sigmoid, sigmoidDerivative
)

print('\n\nThe dervivative deltas for the network are solved as:')
for i, deltaList in enumerate(deltas):
    print('\tLayer '+str(i+1)+': '+str(deltaList))
print('\n')