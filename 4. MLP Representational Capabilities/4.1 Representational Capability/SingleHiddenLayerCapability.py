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
from utils.auxfunctions import plot_decision_boundary


"""
    One-hidden-layer networks can represent nonconvex, disjoint regions.

    Weiland and Leighton provided an example of a nonconvex region that
    can be recognized by a single-hidden-layer ntwork. The example
    creates a "C" shaped region, which is a nonconvex decision region.
"""

class SingleHiddenLayerNetwork:

    def __init__(self, numInputs, hiddenLayerSize, randomize=True, bias=True):
        """
            Create the single hidden network with the above specifications
            to test capability of a single hidden layer network.
        """
        # Set parameters
        self.numInputs = numInputs
        self.hiddenLayerSize = hiddenLayerSize
        self.bias = bias
        # Set up weights for the initial hidden layer
        if bias:
            inputSize = numInputs + 1
        else:
            inputsize = numInputs
        if randomize:
            self.w1 = np.random.rand(hiddenLayerSize, inputSize) * 2.0 - 1.0
        else:
            self.w1 = np.zeros((hiddenLayerSize, inputSize))
        # Set up weights for final layer
        if bias:
            inputSize = hiddenLayerSize + 1
        else:
            inputsize = hiddenLayerSize
        if randomize:
            self.w2 = np.random.rand(1, inputSize) * 2.0 - 10
        else:
            self.w3 = np.zeros((1, inputSize))

    def evaluateHidden(self, inputs):
        """
            Evaluate the output of the hidden layer in the network given the inputs.
        """
        if self.bias:
            inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        output = np.dot(inputs, self.w1.T)
        # Linear thresholding activation
        output[output <= 0] = 0
        output[output > 0] = 1
        return output

    def evaluateFinal(self, inputs):
        """
            Evaluate the output of the final layer in the network given the inputs.
        """
        # Get hidden layer output
        if self.bias:
            inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        hidden_output = np.dot(inputs, self.w1.T)
        # Linear thresholding activation
        hidden_output[hidden_output <= 0] = 0
        hidden_output[hidden_output > 0] = 1
        # Get final output
        if self.bias:
            hidden_output = np.concatenate((hidden_output, np.ones((hidden_output.shape[0], 1))), axis=1)
        output = np.dot(hidden_output, self.w2.T)
        # Linear thresholding activation
        output[output <= 0] = 0
        output[output > 0] = 1
        return output

    
if __name__ == '__main__':
    """
        Testing the decision boundaries for the 'C' shaped region example network.
    """
    shln = SingleHiddenLayerNetwork(2, 10, randomize=False, bias=True)

    # Create pyplot
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('One-Hidden-Layer Networks Can Represent Nonconvex Regions',
                 fontsize=18)

    # Set hard-coded weights to form the example decision boundary
    shln.w1 = np.array([
        [1, 0, -0.2],
        [-1, 0, 0.4],
        [1, 0, -0.6],
        [-1, 0, 0.8],
        [0, 1, -0.1],
        [0, -1, 0.3],
        [0, -1, 0.4],
        [0, 1, -0.6],
        [0, 1, -0.7],
        [0, -1, 0.9]
    ])

    shln.w2 = np.array([
        [2, 1, 0.5, 2, 2, 1, 0.3, 0.3, 1, 2, -8.7]
    ])

    # Plot final layer output
    plot_decision_boundary(shln.evaluateFinal, 0, [0.0, 1.0, 0.0, 1.0],
                           ax, steppingSize=0.01)

    # Add title to ax
    ax.set_title('C-Shaped Region Formed by One-Hidden Layer Net')

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    """
        Testing if a one-hidden-layer network can define disjoint regions.
    """
    shln = SingleHiddenLayerNetwork(2, 3, randomize=False, bias=True)

    # Create pyplot
    fix, ax = plt.subplots(1, 1)
    fig.suptitle('One-Hidden-Layer Networks Can Represent Disjoint Regions',
                 fontsize=18)
    
    # Set hard-coded weights to form the example decision boundary
    shln.w1 = np.array([
        [0, -1, 0],
        [1, 1, -1],
        [-1, 1, 0]
    ])

    shln.w2 = np.array([
        [1, 1, 1, -1.9]
    ])

    # Plot final layer output
    plot_decision_boundary(shln.evaluateFinal, 0, [0.0, 1.0, 0.0, 1.0],
                           ax, steppingSize=0.01)

    # Add title to ax
    ax.set_title('Disjoint Region Formed by One-Hidden Layer Net')

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

