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
    Two hidden layers are sufficient to create classification
    regions of any desired shape. Linear threshold units in the
    first hidden layer divide the input space into half-spaces
    with hyperplanes, units in the second layer AND (form intersections
    of) these half-spaces to produce convex regions, and output
    units OR (form unions of) the convex regions into arbitrary,
    possibly unconnected, shapes.

    The boundaries are piecewise linear, but any smooth boundary can be
    approximated with enough units.
"""

class TwoHiddenLayerNetwork:
    
    def __init__(self, numInputs, hiddenLayerSize1, hiddenLayerSize2,
                 randomize=True, bias=True):
        """
            Create the Two Hidden Layer Network with the above specifications
            to test capability of a two hidden layer network.
        """
        # Set parameters
        self.numInputs = numInputs
        self.hiddenLayerSize1 = hiddenLayerSize1
        self.hiddenLayerSize2 = hiddenLayerSize2
        self.bias = bias
        # Set up weights for initial hidden layer
        if bias:
            inputSize = numInputs + 1
        else:
            inputSize = numInputs
        if randomize:
            self.w1 = np.random.rand(hiddenLayerSize1, inputSize) * 2.0 - 1.0
        else:
            self.w1 = np.zeros((hiddenLayerSize1, inputSize))
        # Set up weights for second hidden layer
        if bias:
            inputSize = hiddenLayerSize1 + 1
        else:
            inputSize = hiddenLayerSize1
        if randomize:
            self.w2 = np.random.rand(hiddenLayerSize2, inputSize) * 2.0 - 1.0
        else:
            self.w2 = np.zeros((hiddenLayerSize2, inputSize))
        # Set up weights for third hidden layer
        if bias:
            inputSize = hiddenLayerSize2 + 1
        else:
            inputSize = hiddenLayerSize2
        if randomize:
            self.w3 = np.random.rand(1, inputSize) * 2.0 - 1.0
        else:
            self.w3 = np.zeros((1, inputSize))
        
    def evaluateHidden1(self, inputs):
        """
            Evaluate the output of the first hidden layer in the network
            using the network's current weights.
        """
        if self.bias:
            inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        output = np.dot(inputs, self.w1.T)
        # Linear thresholding activation
        output[output <= 0] = 0
        output[output > 0] = 1
        return output
        
    def evaluateHidden2(self, inputs):
        """
            Evaluate the output of the second hidden layer in the network
            given the inputs.
        """
        # Get hidden layer 1 output
        hidden1_output = self.evaluateHidden1(inputs)
        # Get hidden layer 2 output
        if self.bias:
            inputs = np.concatenate((hidden1_output, np.ones((hidden1_output.shape[0], 1))), axis=1)
        output = np.dot(inputs, self.w2.T)
        # Linear thresholding activation
        output[output <= 0] = 0
        output[output > 0] = 1
        return output

    def evaluateFinal(self, inputs):
        """
            Evaluate the output of the final layer in the network given
            the inputs
        """
        # Get hidden layer 1 output
        hidden1_output = self.evaluateHidden1(inputs)
        # Get hidden layer 2 output
        hidden2_output = self.evaluateHidden2(inputs)
        # Get final output
        if self.bias:
            inputs = np.concatenate((hidden2_output, np.ones((hidden2_output.shape[0], 1))), axis=1)
        output = np.dot(inputs, self.w3.T)
        # Linear thresholding activation
        output[output <= 0] = 0
        output[output > 0] = 1
        return output


if __name__ == '__main__':
    """
        Testing the decision boundaries at various layers in the network
    """
    thln = TwoHiddenLayerNetwork(2, 3, 2)

    # Create pyplot
    fig, axes = plt.subplots(3, 5)
    fig.suptitle('Decision Boundaries at Various Points in a ' +
                 'Two Hidden Layer Network', fontsize=18)
    
    # Set hard-coded weights to form a triangle with
    # each hyperplane representing an edge
    thln.w1 = np.array([
        [-1, -1, 0],
        [0, 1, 1],
        [1, -1, 0]
    ])

    # First weight vector defines AND
    # Second weight vector is arbitrary
    thln.w2 = np.array([
        [1, 1, 1, -2.5],
        [1, -1, -1, 0]
    ])

    # OR function encoded in weights
    thln.w3 = np.array([
        [1, 1, 0]
    ])

    # Plot hidden layer 1 outputs
    plot_decision_boundary(thln.evaluateHidden1, 0, [-2.0, 2.0, -2.0, 2.0],
                           axes[2, 0], steppingSize=0.01)
    plot_decision_boundary(thln.evaluateHidden1, 1, [-2.0, 2.0, -2.0, 2.0],
                           axes[2, 2], steppingSize=0.01)
    plot_decision_boundary(thln.evaluateHidden1, 2, [-2.0, 2.0, -2.0, 2.0],
                           axes[2, 4], steppingSize=0.01)
    
    # Plot hidden layer 2 outputs
    plot_decision_boundary(thln.evaluateHidden2, 0, [-2.0, 2.0, -2.0, 2.0],
                           axes[1, 1], steppingSize=0.01)
    plot_decision_boundary(thln.evaluateHidden2, 1, [-2.0, 2.0, -2.0, 2.0],
                           axes[1, 3], steppingSize=0.01)

    # Plot final layer output
    plot_decision_boundary(thln.evaluateFinal, 0, [-2.0, 2.0, -2.0, 2.0],
                           axes[0, 2], steppingSize=0.01)

    # Delete the unused axes
    unusedAxesFirstRow = [axes[0, i] for i in [0, 1, 3, 4]]
    for ax in unusedAxesFirstRow:
        fig.delaxes(ax)
    unusedAxesSecondRow = [axes[1, i] for i in [0, 2, 4]]
    for ax in unusedAxesSecondRow:
        fig.delaxes(ax)
    unusedAxesThirdRow = [axes[2, i] for i in [1, 3]]
    for ax in unusedAxesThirdRow:
        fig.delaxes(ax)

    # Add titles to axes
    for i in range(3):
        axes[2, i*2].set_title('Hidden Layer 1\nNode ' + str(i + 1) + ' Output')
    axes[1, 1].set_title('Hidden Layer 2\nNode 1 Output\n(Simulating AND)')
    axes[1, 3].set_title('Hidden Layer 2\nNode 2 Output\n(Arbitrary Weights)')
    axes[0, 2].set_title('Final Output\n(Simulating OR)')

    # Set caption
    textstr = str('Three layers of linear threshold units are sufficient ' +
                  'to define arbitrary regions.\nUnits in the first hidden ' +
                  'layer divide the input space with hyperplanes, units ' +
                  'in\nthe second hidden layer can form convex regions bounded ' +
                  'by these hyperplanes,\nand output units can combine the ' +
                  'regions defined by the second layer into\narbitrarily shaped, ' +
                  'possibly unconnected, regions.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    plt.text(-30, 12, textstr, fontsize=13, bbox=props)

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()