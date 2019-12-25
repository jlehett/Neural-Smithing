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

    def trainWithMomentum(self, trainType, inputs, targetOutputs, 
                          alpha, learningRate, epochs, verbose=True):
        """
            Train the network with momentum using the specified training type
            ('batch' or 'online'). Momentum is controlled by the alpha parameter.
        """
        # Iterate through each epoch of training
        prevWeightChange = None
        for e in range(epochs):
            # Train for 1 epoch depending on type and grab the weight change
            newWeightChange = None
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
            # Add the fraction of the previous weight change
            if prevWeightChange != None:
                for weightLayerIndex in range(len(self.weights)):
                    self.weights[weightLayerIndex] += alpha * prevWeightChange[weightLayerIndex]
            # Set the previous weight change to the current one that was just produced
            prevWeightChange = newWeightChange
            for weightLayerIndex in range(len(self.weights)):
                prevWeightChange[weightLayerIndex] += alpha * prevWeightChange[weightLayerIndex]
        # Print out the final metrics after training
        acc, loss = self.getMetrics(inputs, targetOutputs)
        print(
            '\nFinal Loss: {0:.4f}'.format(loss) +
            '\tFinal Acc: {0:.4f}'.format(acc)
        )



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
    network.trainWithMomentum(
        'batch', inputs, targetOutputs, alpha=1.75, learningRate=0.1, epochs=1000, verbose=True
    )

    print('\n\nThe target outputs are:\n')
    print(np.asarray(targetOutputs))

    print('\n\nThe network outputs are:\n')
    print(
        network.feedforward(
            inputs
        )
    )