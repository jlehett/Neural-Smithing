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

from PerceptronLearningAlgorithm import PerceptronLearningNetwork
# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import plotModel


"""
    The perceptron algorithm is guaranteed to learn pattern sets that are
    linearly separable. However, it does not converge when the training
    data is not linearly separable. It is possible for the system to visit
    a near optimal set of weights, and then wander away to a very bad set.

    Gallant's "pocket algorithm" keeps a copy of the best weight vector
    obtained so far. The quality of the pocket weights thus tends to
    improve over time and converge to an optimum set.
"""

class PocketAlgorithmNetwork(PerceptronLearningNetwork):

    def __init__(self, numInputs, numOutputs, randomize=False, bias=False):
        """
            Construct a single-layer neural network given the number of input
            nodes and the number of output nodes. This network will train
            using the pocket algorithm. The activation function used in 
            this network will be a linear threshold unit where
            y = -1 if u <= 0
            y = +1 if u > 0     where u is the output of the weighted sum of inputs

            Input, output, and target values are assumed to be +- 1
        """
        # Initialize like it is the standard perceptron learning algorithm
        super().__init__(numInputs, numOutputs, randomize, bias)
        # Create a "pocket" set of weights that tracks the best set
        self.best_accuracy = 0.0
        self.best_weights = np.zeros(self.weights.shape)

    def train(self, x, y, epochs=1, lr=0.1, verbose=True):
        """
            Train the network given a batch of inputs, x, and their corresponding
            target outputs, y. Run the pocket algorithm on the training
            set for the number of times specified by the epochs parameter.
        """
        numTrainingPoints = len(x)
        for e in range(epochs):
            # Set accuracy at beginning of epoch to 0s
            accuracy = 0
            # Compute the output for all training points
            allOutputs = self.evaluate(x, training=True)
            for i in range(numTrainingPoints):
                # Grab the inputs for the specific training point
                trainingPointInputs = x[i]
                # Grab the output for the specific training point
                trainingPointOutput = allOutputs[i]
                # Get the target outputs for the specific training point
                targets = y[i]
                # Compare each output 1 by 1
                for outputIndex in range(len(trainingPointOutput)):
                    # Grab specific output and corresponding target value
                    targetVal = targets[outputIndex]
                    outputVal = trainingPointOutput[outputIndex]
                    # If the outputs match, increment accuracy
                    if targetVal == outputVal:
                        accuracy += 1
                        continue
                    # Else, update the weights
                    else:
                        # For each input weight, compute its delta change, and the apply the change
                        for inputWeightIndex in range(len(self.weights[outputIndex])):
                            # If the inputWeightIndex is in the range of values for inputs, use the input at that index
                            if inputWeightIndex < len(trainingPointInputs):
                                trainingPointInput = trainingPointInputs[inputWeightIndex]
                            # Else, that value is the bias, and the input should be constant 1.0
                            else:
                                trainingPointInput = 1.0
                            # Compute delta w and apply the change
                            deltaW = 2.0 * lr * targetVal * trainingPointInput
                            self.weights[outputIndex, inputWeightIndex] += deltaW
            # Compute accuracy
            accuracy /= numTrainingPoints
            # If new best accuracy is obtained, update pocket weights
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_weights = np.copy(self.weights)
            # If verbose == True, print accuracy for each training epoch
            if verbose:
                print('Epoch ' + str(e+1) + ' / ' + str(epochs) + ' Accuracy: ' + str(accuracy))
            # Return the final accuracy
        return self.best_accuracy

    def evaluate(self, inputs, training=False):
        """
            Evaluate the inputs passing through the neural network
            using the network's pocket (best) weights.
        """
        if self.bias:
            inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        if training:
            self.outputNodes = np.dot(inputs, self.weights.T)
        else:
            self.outputNodes = np.dot(inputs, self.best_weights.T)
        for i in range(len(self.outputNodes)):
            self.outputNodes[i] = self.activationFunc(self.outputNodes[i])
        return self.outputNodes

    def getWeights(self):
        """
            Return the networks current weights in a numpy array of size
            ( self.outputNodes.shape[0], self.inputNodes.shape[0] )
        """
        return self.best_weights
        


if __name__ == '__main__':
    """"
        Test the Pocket Algorithm and Perceptron Learning Algorithm on
        the XOR dataset.
    """
    # Params
    EPOCHS = 50
    NUM_GRAPHS = 10

    NUM_EPOCHS_PER_GRAPH = math.ceil(EPOCHS / NUM_GRAPHS)
    # Create XOR training data
    xor_train_x = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])
    xor_train_y = np.array([
        [-1],
        [1],
        [1],
        [-1]
    ])
    # Create networks
    xor_pan = PocketAlgorithmNetwork(2, 1, randomize=True, bias=True)
    xor_pln = PerceptronLearningNetwork(2, 1, randomize=True, bias=True)

    # Create the suplots
    fig, axes = plt.subplots(2, NUM_GRAPHS)
    fig.suptitle('Comparing Pocket Algorithm and Perceptron Learning Algorithm\n' +
                 'on XOR Boolean function.', fontsize=18)

    # Label the subplots
    for x in range(2):
        for y in range(1, NUM_GRAPHS+1):
            if x == 0:
                axes[x][y-1].set_title('Pocket\nAlgorithm at\n' + str(y * int(NUM_EPOCHS_PER_GRAPH)) + ' Epochs')
            else:
                axes[x][y-1].set_title('Perceptron\nLearning\nAlgorithm at\n' + str(y * int(NUM_EPOCHS_PER_GRAPH)) + ' Epochs')
    
    # Train each model on their respective datasets and show progress in the form
    # of decision boundaries for every N epochs
    for i in range(NUM_GRAPHS):
        # Train each model for N more epochs
        xor_pan.train(xor_train_x, xor_train_y, epochs=NUM_EPOCHS_PER_GRAPH, lr=0.1, verbose=False)
        xor_pln.train(xor_train_x, xor_train_y, epochs=NUM_EPOCHS_PER_GRAPH, lr=0.1, verbose=False)

        # Plot the model
        plotModel(xor_pan, axes[0, i], [-1.2, 1.2, -1.2, 1.2], [-1.0, 1.0])
        axes[0, i].set_xlim(-1.2, 1.2)
        axes[0, i].set_ylim(-1.2, 1.2)
        axes[0, i].set_aspect('equal', adjustable='box')

        plotModel(xor_pln, axes[1, i], [-1.2, 1.2, -1.2, 1.2], [-1.0, 1.0])
        axes[1, i].set_xlim(-1.2, 1.2)
        axes[1, i].set_ylim(-1.2, 1.2)
        axes[1, i].set_aspect('equal', adjustable='box')

        # Plot the actual data for each model
        axes[0, i].scatter([-1, 1], [1, -1], c='blue', s=150)
        axes[0, i].scatter([-1, 1], [-1, 1], c='red', s=150)
        
        axes[1, i].scatter([-1, 1], [1, -1], c='blue', s=150)
        axes[1, i].scatter([-1, 1], [-1, 1], c='red', s=150)

    # Set caption
    textstr = str('Notice how the Pocket Algorithm only changes when\n' +
                  'it discovers a set of weights that produces a more\n' +
                  'accurate decision boundary.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    plt.text(-17, 4, textstr, fontsize=14, bbox=props)

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


    """
        Let's compare how Pocket Algorithm performs on a randomized dataset
        compared to the Perceptron Learning Algorithm on the same dataset.
    """
    # Params
    NUM_DATAPOINTS = 500
    NUM_DIMS = 25
    NUM_EPOCHS = 50

    # Create random dataset
    train_x = np.zeros((500, 25))
    train_y = np.zeros((500, 1))
    for datapointIndex in range(NUM_DATAPOINTS):
        if random.randint(0, 1) == 0:
            train_y[datapointIndex, 0] = -1
        else:
            train_y[datapointIndex, 0] = 1
        for dimIndex in range(NUM_DIMS):
            if random.randint(0, 1) == 0:
                train_x[datapointIndex, dimIndex] = -1
            else:
                train_x[datapointIndex, dimIndex] = 1
    
    # Create networks
    pan = PocketAlgorithmNetwork(NUM_DIMS, 1, randomize=True, bias=True)
    pln = PerceptronLearningNetwork(NUM_DIMS, 1, randomize=True, bias=True)

    # Create the plot
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Comparing Pocket Algorithm and Perceptron Learning Algorithm\n' +
                 'on a randomized dataset', fontsize=18)
    
    # Train the networks on the training data for n epochs and record
    # the accuracy at the end of each epoch
    pan_acc = []
    pln_acc = []
    for _ in range(NUM_EPOCHS):
        pan_acc.append(pan.train(train_x, train_y, epochs=1, lr=0.1, verbose=False))
        pln_acc.append(pln.train(train_x, train_y, epochs=1, lr=0.1, verbose=False))

    # Plot the accuracies of each network over time
    ax.plot(range(1, NUM_EPOCHS+1), pan_acc, color='blue', label='Pocket Algorithm')
    ax.plot(range(1, NUM_EPOCHS+1), pln_acc, color='red', label='Perceptron Learning Algorithm')

    # Set legend for the plot
    fig.legend(loc='upper right', borderaxespad=1.0, fontsize=15)

    # Set caption
    textstr = str('Notice how the Pocket Algorithm\'s performance\n' +
                  'strictly increases, while the Perceptron Learning\n' +
                  'algorithm never converges.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    plt.text(25, 0.2, textstr, fontsize=14, bbox=props)

    # Adjust graph parameters
    ax.grid(linewidth=1, alpha=0.5)
    ax.set_xlim(-0.3, 50.3)
    ax.set_ylim(0.0, 1.0)

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
