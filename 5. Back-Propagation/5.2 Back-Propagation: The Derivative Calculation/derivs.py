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
sys.path.append('../5.1 Preliminaries/')
# Import any python modules from parent directory
from utils.auxfunctions import MultiLayerNetwork
from utils.auxfunctions import sigmoid, sigmoidDerivative
from prelims import SSEFunctionNetwork

"""
    Having obtained the outputs and calculated the error, the next
    step is to calculate the derivative of the error with respect
    to the weights.

    The total derivative is just the sum of the per-pattern derivatives.
"""

# Create a subclass that adds a function to find weight derivatives to the
# multi-layer network class we've been building up in section 5.1
class GetDerivsNetwork(SSEFunctionNetwork):

    def getDerivs(self, inputs, targetOutputs):
        # Set up total derivatives
        totalDerivatives = []
        # Iterate through each input to obtain the gradient to add to
        # totalDerivative
        for inputNum, input_ in enumerate(inputs[:1]):
            networkOutput = self.feedforward([input_])
            # Setup lists to hold deltas and derivs
            deltas = []
            derivs = []
            # Find the deltas for the last layer of weights
            print(targetOutputs[inputNum])
            print(networkOutput[0])
            deltas.append(
                -(targetOutputs[inputNum] - networkOutput[0]) * 
                self.activationDerivative(self.outputNodes[0])
            )
            # Find the deltas for the rest of the layers
            for i in range(-1, -len(self.hiddenNodes)-1, -1):
                deltas.append(
                    self.activationDerivative(self.hiddenNodes[i][0]) *
                    np.dot(self.weights[i], deltas[-1])
                )
            # Reverse the list of deltas such that the deltas in the earlier
            # layers are found in the earlier indices of the list
            deltas.reverse()
            # Find the derivatives of the network
            derivs.append(
                np.outer(self.activationFunction(self.inputNodes[0]), deltas[0])
            )
            for hiddenIndex in range(len(self.hiddenNodes)):
                derivs.append(
                    np.outer(
                        self.activationFunction(self.hiddenNodes[hiddenIndex])[0], 
                        deltas[hiddenIndex+1]
                    )
                )
            # Either append the derivative to the totalDerivatives if this
            # is the first input back-propagated, or add to the existing
            # totalDerivatives otherwise
            if inputNum == 0:
                totalDerivatives = derivs
            else:
                for derivIndex in range(len(totalDerivatives)):
                    totalDerivatives[derivIndex] += derivs[derivIndex]
        # Return the derivatives at the end
        return totalDerivatives

if __name__ == '__main__':
    # Construct the network
    network = GetDerivsNetwork(
        2, [3, 3], 2, sigmoid, sigmoidDerivative, bias=True, randomize=True
    )

    # Get the derivatives for the network on the given input
    derivs = network.getDerivs(
        [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ],
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 1]
        ]
    )

    # Print out the results
    print('\n\nThe network derivatives were calculated as:\n')
    print('1st set of weights:')
    print(derivs[0])
    print('\n2nd set of weights:')
    print(derivs[1])
    """
    print('\n3rd set of weights:')
    print(derivs[2])
    print('\n\n')
    """