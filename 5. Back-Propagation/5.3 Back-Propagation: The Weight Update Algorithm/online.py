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

    In on-line learning, the weights are updated after each pattern
    presentation. Generally, a pattern p is chosen at random and
    presented to the network. The output is compared with the target for
    that pattern and the errors are back-propagated to obtain the
    single-pattern derivative. The weights are then updated immediately,
    using the gradient of the single-pattern error.
"""

# Create a subclass that adds a function to fully perform on-line learning
# to the multi-layer network class we've been building up in sections 
# 5.1 & 5.2
class OnlineNetwork(GetDerivsNetwork):

    def onlineLearning(self, inputs, targetOutputs, learningRate, epochs, verbose=True,
                       printFinal=True, printNum=None, totalNum=None):
        """
            Perform on-line learning on the inputs with corresponding target outputs.
            printFinal, printNum, and totalNum should stay True; they are used in morphing
            this function to fit the momentum training requirements.
        """
        # Iterate through loop on each epoch
        derivatives = None
        for e in range(epochs):
            # Grab the random index to use in selecting the pattern to train on for this epoch.
            randomIndex = random.randint(0, len(inputs) - 1)
            # First, obtain the derivatives via the getDerivs function that was built in
            # section 5.2
            derivatives = self.getDerivs([inputs[randomIndex]], [targetOutputs[randomIndex]])
            # Update all of the weights
            for weightLayerIndex in range(len(self.weights)):
                self.weights[weightLayerIndex] += -learningRate * derivatives[weightLayerIndex]
            # If verbose, print out the training progress
            if verbose:
                networkOutputs = self.feedforward(inputs)
                acc, loss = self.getMetrics(inputs, targetOutputs)
                # Control what epoch number is printed depending on momentum training
                epochNum = e
                if printNum:
                    epochNum = printNum
                totalEpochs = epochs
                if totalNum:
                    totalEpochs = totalNum
                # Print training progress
                print(
                    'Epoch ' + str(epochNum+1) + ' / ' + str(totalEpochs) + ':' +
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
        # Return the last weight change value
        for d in range(len(derivatives)):
            derivatives[d] *= -learningRate
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
    network = OnlineNetwork(
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