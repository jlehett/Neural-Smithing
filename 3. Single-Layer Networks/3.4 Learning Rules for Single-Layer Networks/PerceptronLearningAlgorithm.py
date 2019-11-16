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
from utils.SingleLayerNetwork import SingleLayerNetwork
from utils.auxfunctions import plotModel


"""
    The perceptron learning algorithm is suitable for learning linearly
    separable binary functions of binary inputs and is guaranteed to 
    find a solution if the classes are linearly separable.

    Unfortunately, if the classes are not linearly separable, the 
    algorithm may not even converge and is unlikely to produce the best
    solution when it does.
"""

class PerceptronLearningNetwork(SingleLayerNetwork):

    def __init__(self, numInputs, numOutputs, randomize=False, bias=False):
        """
            Construct a single-layer neural network given the number of input
            nodes and the number of output nodes. This network will train
            using the perceptron learning algorithm. The activation function
            used in this network will be a linear threshold unit where
            y = -1 if u <= 0
            y = +1 if u > 0     where u is the output of the weighted sum of inputs

            Input, output, and target values are assumed to be +- 1
        """
        # Define activation function
        def ltu(x):
            if x <= 0:
                return -1
            return 1
        # Parameters
        self.bias = bias
        self.activationFunc = ltu
        # Set weights according to randomize parameter
        if bias:
            self.weights = np.zeros((numOutputs, numInputs+1))
        else:
            self.weights = np.zeros((numOutputs, numInputs))
        if randomize:
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    self.weights[i, j] = random.random() * 2.0 - 1.0

    def train(self, x, y, epochs=1, lr=0.1, verbose=True):
        """
            Train the network given a batch of inputs, x, and their corresponding
            target outputs, y. Run the perceptron learning algorithm on the training
            set for the number of times specified by the epochs parameter.
        """
        numTrainingPoints = len(x)
        for e in range(epochs):
            # Set accuracy at beginning of epoch to 0s
            accuracy = 0
            # Compute the output for all training points
            allOutputs = self.evaluate(x)
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
            # If verbose == True, print accuracy for each training epoch
            if verbose:
                print('Epoch ' + str(e+1) + ' / ' + str(epochs) + ' Accuracy: ' + str(accuracy))
            # Return final accuracy
        return accuracy


# Test the network's learning capabilities
if __name__ == '__main__':
    """"
        Test the network on the AND and XOR Boolean functions. Show training 
        progress in graphs.
    """
    # Params
    EPOCHS = 10
    NUM_GRAPHS = 10

    NUM_EPOCHS_PER_GRAPH = math.ceil(EPOCHS / NUM_GRAPHS)
    # Create AND training data
    and_train_x = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])
    and_train_y = np.array([
        [-1],
        [-1],
        [-1],
        [1]
    ])
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
    and_pln = PerceptronLearningNetwork(2, 1, randomize=True, bias=True)
    xor_pln = PerceptronLearningNetwork(2, 1, randomize=True, bias=True)

    # Create the suplots
    fig, axes = plt.subplots(2, NUM_GRAPHS)
    fig.suptitle('Perceptron Learning Algorithm Training on AND and XOR Boolean Functions',
                 fontsize=18)

    # Label the subplots
    for x in range(2):
        for y in range(1, NUM_GRAPHS+1):
            if x == 0:
                axes[x][y-1].set_title('XOR Model at\n' + str(y * int(NUM_EPOCHS_PER_GRAPH)) + ' Epochs')
            else:
                axes[x][y-1].set_title('AND Model at\n' + str(y * int(NUM_EPOCHS_PER_GRAPH)) + ' Epochs')
    
    # Train each model on their respective datasets and show progress in the form
    # of decision boundaries for every N epochs
    for i in range(NUM_GRAPHS):
        # Train each model for N more epochs
        xor_pln.train(xor_train_x, xor_train_y, epochs=NUM_EPOCHS_PER_GRAPH, lr=0.05, verbose=False)
        and_pln.train(and_train_x, and_train_y, epochs=NUM_EPOCHS_PER_GRAPH, lr=0.05, verbose=False)

        # Plot the model
        plotModel(xor_pln, axes[0, i], [-1.2, 1.2, -1.2, 1.2], [-1.0, 1.0])
        axes[0, i].set_xlim(-1.2, 1.2)
        axes[0, i].set_ylim(-1.2, 1.2)
        axes[0, i].set_aspect('equal', adjustable='box')

        plotModel(and_pln, axes[1, i], [-1.2, 1.2, -1.2, 1.2], [-1.0, 1.0])
        axes[1, i].set_xlim(-1.2, 1.2)
        axes[1, i].set_ylim(-1.2, 1.2)
        axes[1, i].set_aspect('equal', adjustable='box')

        # Plot the actual data for each model
        axes[0, i].scatter([-1, 1], [1, -1], c='blue', s=150)
        axes[0, i].scatter([-1, 1], [-1, 1], c='red', s=150)
        
        axes[1, i].scatter([1], [1], c='blue', s=150)
        axes[1, i].scatter([-1, -1, 1], [-1, 1, -1], c='red', s=150)

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


    """
        14 of the 16 Boolean functions are linearly separable.
        Let's show all of them and how Single-Layer Networks perform when
        using the Perceptron Learning Algorithm.
    """
    # Params
    EPOCHS_PER_MODEL = 20

    # Define class to hold info about each Boolean function
    class BooleanFunction:
        def __init__(self, true_x, false_x, name):
            self.true_x = true_x
            self.false_x = false_x
            self.name = name
        
        def getTrainingData(self):
            training_x = []
            training_y = []
            for pair in self.true_x:
                training_x.append(pair)
                training_y.append([1])
            for pair in self.false_x:
                training_x.append(pair)
                training_y.append([-1])
            return np.array(training_x), np.array(training_y)

    # Define each of the 16 boolean functions
    functions = []
    functions.append(
        BooleanFunction(
            [],
            [[-1, -1], [-1, 1], [1, -1], [1,1]],
            'FALSE'
        )
    )
    functions.append(
        BooleanFunction(
            [[1, 1]],
            [[-1, -1], [-1, 1], [1, -1]],
            'AND'
        )
    )
    functions.append(
        BooleanFunction(
            [[1, -1]],
            [[-1, -1], [-1, 1], [1, 1]],
            'A AND NOT B'
        )
    )
    functions.append(
        BooleanFunction(
            [[1, -1], [1, 1]],
            [[-1, -1], [-1, 1]],
            'A'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, 1]],
            [[-1, -1], [1, -1], [1, 1]],
            'NOT A AND B'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, 1], [1, 1]],
            [[-1, -1], [1, -1]],
            'B'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, 1], [1, -1]],
            [[-1, -1], [1, 1]],
            'XOR'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, 1], [1, -1], [1, 1]],
            [[-1, -1]],
            'OR'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1]],
            [[-1, 1], [1, -1], [1, 1]],
            'NOR'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1], [1, 1]],
            [[-1, 1], [1, -1]],
            'XNOR'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1], [1, -1]],
            [[-1, 1], [1, 1]],
            'NOT B'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1], [1, -1], [1, 1]],
            [[-1, 1]],
            'A OR NOT B'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1], [-1, 1]],
            [[1, -1], [1, 1]],
            'NOT A'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1], [-1, 1], [1, 1]],
            [[1, -1]],
            'NOT A OR B'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1], [-1, 1], [1, -1]],
            [[1, 1]],
            'NAND'
        )
    )
    functions.append(
        BooleanFunction(
            [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            [],
            'TRUE'
        )
    )

    # Create the pyplot
    fig, axes = plt.subplots(2, 8)
    fig.suptitle('14 of 16 Boolean functions are linearly separable.\n' +
                'These graphs display decision boundaries formed by training\n' +
                'Single-Layer Networks using Perceptron Learning Algorithm.',
                fontsize=14)

    # Print out a warning to user that this may take a while, but progress will be output below
    print('\n\nWARNING: This may take a while. Progress will be output below:\n\n')

    # Create models for each of the functions, and plot the trained models
    f, x, y = 0, 0, 0
    for function in functions:
        # Create Single Layer Network
        model = PerceptronLearningNetwork(2, 1, randomize=True, bias=True)

        # Train the model on its dataset
        training_x, training_y = function.getTrainingData()
        model.train(training_x, training_y, epochs=EPOCHS_PER_MODEL, lr=0.5, verbose=False)

        # Print progress
        print('PROGRESS: ' + str(f+1) + ' / 16')

        # Plot the trained model in the appropriate subplot
        plotModel(model, axes[x, y], [-1.2, 1.2, -1.2, 1.2], valRange=[-1.0, 1.0])
        axes[x, y].set_xlim(-1.2, 1.2)
        axes[x, y].set_ylim(-1.2, 1.2)
        axes[x, y].set_aspect('equal', adjustable='box')
        axes[x, y].set_title(function.name)

        # Plot the points in the dataset
        for pair in function.true_x:
            axes[x, y].scatter(pair[0], pair[1], color='blue', s=150)
        for pair in function.false_x:
            axes[x, y].scatter(pair[0], pair[1], color='red', s=150)


        # Increment the trackers
        f += 1
        x += 1
        if x >= 2:
            x = 0
            y += 1

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()