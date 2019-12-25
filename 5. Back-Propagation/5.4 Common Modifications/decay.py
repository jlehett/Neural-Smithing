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
class DecayNetwork(OnlineNetwork, BatchNetwork):

    def trainWithDecay(self, trainType, inputs, targetOutputs, 
                       rho, learningRate, epochs, verbose=True):
        """
            Train the network with decay using the specified training type
            ('batch' or 'online'). Decay is controlled by the rho parameter.
        """
        # Iterate through each epoch of training
        for e in range(epochs):
            # Grab previous weights
            prevWeights = self.weights
            # Train for 1 epoch depending on type
            if trainType == 'batch':
                newWeightChange = self.batchLearning(
                    inputs, targetOutputs, learningRate, 1,
                    verbose=True, printFinal=False, printNum=e,
                    totalNum=epochs
                )
            if trainType == 'online':
                newWeightChange = self.onlineLearning(
                    inputs, targetOutputs, learningRate, 1,
                    verbose=True, printFinal=False, printNum=e,
                    totalNum=epochs
                )
            # Decay the weights by subtracting by a fraction of the current weights
            for weightLayerIndex in range(len(prevWeights)):
                prevWeights[weightLayerIndex] -= rho * prevWeights[weightLayerIndex]
                prevWeights[weightLayerIndex] += newWeightChange[weightLayerIndex]
            self.weights = prevWeights
        # Print out the final metrics after training
        acc, loss = self.getMetrics(inputs, targetOutputs)
        print(
            '\nFinal Loss: {0:.4f}'.format(loss)
        )



if __name__ == '__main__':
    # Construct the network
    network = DecayNetwork(
        2, [10, 10, 10, 10], 1, sigmoid, sigmoidDerivative, bias=True, randomize=True
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
    network.trainWithDecay(
        'batch', inputs, targetOutputs, rho=0.00005, learningRate=1.0, epochs=1000, verbose=True
    )

    # Find irrelevant weights and change them to 0.0
    numIrrelevantWeights = 0

    for weightLayerIndex in range(len(network.weights)):
        for w1Index in range(len(network.weights[weightLayerIndex])):
            for w2Index in range(len(network.weights[weightLayerIndex][w1Index])):
                if network.weights[weightLayerIndex][w1Index][w2Index] < 0.02 and network.weights[weightLayerIndex][w1Index][w2Index] > -0.02:
                    network.weights[weightLayerIndex][w1Index][w2Index] = 0.0
                    numIrrelevantWeights += 1

    print('\n\nThe target outputs are:\n')
    print(np.asarray(targetOutputs))

    print('\n\nThe network outputs are:\n')
    print(
        network.feedforward(
            inputs
        )
    )

    print('\n\nThe number of irrelevant weights found are: ' + str(numIrrelevantWeights))

    # Train the network
    network.batchLearning(
        inputs, targetOutputs, learningRate=1.0, epochs=1000, verbose=True
    )

    # Find irrelevant weights and change them to 0.0
    numIrrelevantWeights = 0

    for weightLayerIndex in range(len(network.weights)):
        for w1Index in range(len(network.weights[weightLayerIndex])):
            for w2Index in range(len(network.weights[weightLayerIndex][w1Index])):
                if network.weights[weightLayerIndex][w1Index][w2Index] < 0.02 and network.weights[weightLayerIndex][w1Index][w2Index] > -0.02:
                    network.weights[weightLayerIndex][w1Index][w2Index] = 0.0
                    numIrrelevantWeights += 1

    print('\n\nThe target outputs are:\n')
    print(np.asarray(targetOutputs))

    print('\n\nThe network outputs are:\n')
    print(
        network.feedforward(
            inputs
        )
    )

    print('\n\nThe number of irrelevant weights found are: ' + str(numIrrelevantWeights))