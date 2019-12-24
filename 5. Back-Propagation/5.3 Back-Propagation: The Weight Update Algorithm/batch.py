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
sys.path.append('../5.2 Back-Propagation: The Derivative Calculation/')
# Import any python modules from parent directory
from utils.auxfunctions import MultiLayerNetwork
from utils.auxfunctions import sigmoid, sigmoidDerivative
from derivs import GetDerivsNetwork

"""
    Having obtained the outputs and calculated the error, the next
    step is to calculate the derivative of the error with respect
    to the weights.

    The total derivative is just the sum of the per-pattern derivatives.

    In batch training, every pattern p is evaluated to obain the
    derivative terms; these are summed to obtain the total derivative.
"""

# Create a subclass that adds a function to fully perform batch learning
# to the multi-layer network class we've been building up in sections 
# 5.1 & 5.2
class BatchNetwork(GetDerivsNetwork):

    def batchLearning(self, inputs, targetOutputs, learningRate, epochs, verbose=True, printFinal=True):
        # Iterate through loop on each epoch
        derivatives = None
        for e in range(epochs):
            # First, obtain the derivatives via the getDerivs function that was built in
            # section 5.2
            derivatives = self.getDerivs(inputs, targetOutputs)
            # Update all of the weights
            for weightLayerIndex in range(len(self.weights)):
                self.weights[weightLayerIndex] += -learningRate * derivatives[weightLayerIndex]
            # If verbose, print out the training progress
            if verbose:
                networkOutputs = self.feedforward(inputs)
                acc, loss = self.getMetrics(inputs, targetOutputs)
                print(
                    'Epoch ' + str(e+1) + ' / ' + str(epochs) + ':' +
                    '\tLoss: {0:.4f}'.format(loss) + 
                    '\tAcc: {0:.4f}'.format(acc)
                )
        if printFinal:
            # Print out the final metrics after training
            acc, loss = self.getMetrics(inputs, targetOutputs)
            print(
                '\nFinal Loss: {0:.4f}'.format(loss) +
                '\tFinal Acc: {0:.4f}'.format(acc)
            )
        # Return the derivatives
        return derivatives
        
    def getMetrics(self, inputs, targetOutputs):
        # Add a function to obtain accuracy and loss of the network to test if 
        # batch learning is working properly.
        correct = 0
        total = 0
        # Grab network outputs
        networkOutputs = self.feedforward(inputs)
        for i, networkOutput in enumerate(networkOutputs):
            if networkOutput == targetOutputs[i]:
                correct += 1
            total += 1
        # Determine accuracy
        acc = correct / total
        # Grab loss value via SSE function built in section 5.1
        loss = self.SSE(inputs, targetOutputs)
        return acc, loss


if __name__ == '__main__':
    # Construct the network
    network = BatchNetwork(
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
    network.batchLearning(inputs, targetOutputs, 1.0, 1000, verbose=True)

    print('\n\nThe target outputs are:\n')
    print(np.asarray(targetOutputs))

    print('\n\nThe network outputs are:\n')
    print(
        network.feedforward(
            inputs
        )
    )