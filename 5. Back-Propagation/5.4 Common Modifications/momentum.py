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
sys.path.append('../5.3 Back-Propagation: The Weight Update Algorithm/')
# Import any python modules from parent directory
from utils.auxfunctions import MultiLayerNetwork
from utils.auxfunctions import sigmoid, sigmoidDerivative
from online import OnlineNetwork
from batch import BatchNetwork

"""
    A common modification of the basic weight update rule is the addition of a
    momentum term. The idea is to stabilize the weight trajectory by making
    the weight change a combination of the gradient-decreasing term plus a
    fraction of the previous weight change.
"""

# Create a subclass that adds a function to fully perform on-line learning
# to the multi-layer network class we've been building up in sections 
# 5.1 & 5.2
class MomentumNetwork(OnlineNetwork, BatchNetwork):

    def trainWithMomentum(trainType='batch', inputs, targetOutputs, 
                          learningRate, epochs, verbose=True):
        """
            Train the network with momentum using the specified training type
            ('batch' or 'online')
        """
        pass


if __name__ == '__main__':
    # Construct the network
    network = MomentumNetwork(
        2, [4, 4], 1, sigmoid, sigmoidDerivative, bias=True, randomize=True
    )

    # Construct inputs for the network
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]

    # Construct target outputs
    targetOutputs = [
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ]

    # Train the network
    network.onlineLearning(inputs, targetOutputs, 1.0, 4000, verbose=True)

    print('\n\nThe target outputs are:\n')
    print(np.asarray(targetOutputs))

    print('\n\nThe network outputs are:\n')
    print(
        network.feedforward(
            inputs
        )
    )