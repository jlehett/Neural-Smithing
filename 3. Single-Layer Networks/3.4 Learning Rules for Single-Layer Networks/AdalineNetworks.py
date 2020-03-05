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
    Adaline (adaptive linear neuron) networks are a single-layer network
    with a step function as the nonlinearity. Although the perceptron was
    analyzed assuming binary inputs, Adaline inputs may be continuous.

    Weights are adjusted with thhe Widrow-Hoff learning rule to minimize
    the difference between the output and externally supplied target.
    
    the Widrow-Hoff learning rule, also called the LMS algorithm or the
    delta rule, is basically an iterative implementation of linear
    regression. Both minimize the mean squared error of a linear fit.
"""

class AdalineNetwork(PerceptronLearningNetwork):

    def __init__(self, numInputs, numOutputs, randomize=False, bias=False):
        """
            Construct a single-layer neural network given the number of input
            nodes and the number of output nodes. This network will train
            using the Widrow-Hoff learning rule. The activation function
            used in this network will be a linear threshold unit where
            y = -1 if u <= 0
            y = +1 if u > 0     where u is the output of the weighted sum of inputs

            Input, output, and target values are assumed to be +- 1
        """
        super().__init__(numInputs, numOutputs, randomize, bias)
        self.iterations = 0

    def getOutputNoActivation(self, inputs):
        """
            Evaluate the inputs passing through the neural network using the
            network's current weights. Do not pass the outputs through the
            activation function.
        """
        if self.bias:
            inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        self.outputNodes = np.dot(inputs, self.weights.T)
        return self.outputNodes

    def train(self, x, y, epochs=1, verbose=False):
        """
            Train the network given a batch of inputs, x, and their corresponding
            target outputs, y. Run the perceptron learning algorithm on the training
            set for the number of times specified by the epochs parameter.

            The error is reduced by a factor of lr each time the weights are updated.
            Stability requires that 0 < lr < 2 and generally 0.1 < lr < 1.0. For lr=1,
            the error on the present pattern is completely corrected in one cycle;
            for lr > 1, it is overcorrected.

            If the input patterns are linearly independent, the weights will converge
            to unique values. If not, the corrections will be oscillatory and lr should
            decrease over time to allow the weights to settle. One possible schedule is
            lr = k^-1, where k indexes the iterations.
        """
        numTrainingPoints = len(x)
        for e in range(epochs):
            # Set accuracy at beginning of epochs to 0s
            accuracy = 0
            # Compute the output for all training points
            allOutputs = self.getOutputNoActivation(x)
            for i in range(numTrainingPoints):
                # Increment iterations for learning rate scheduling
                self.iterations += 1
                # Calculate the new learning rate from scheduling
                lr = self.iterations ** -1
                # Grab the input for the specific training point
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
                        # For each input weight, compute its delta change, and then apply the change
                        for inputWeightIndex in range(len(self.weights[outputIndex])):
                            # If the inputWeightIndex is in the range of values for inputs, use the input at that index
                            if inputWeightIndex < len(trainingPointInputs):
                                trainingPointInput = trainingPointInputs[inputWeightIndex]
                            # Else, that value is the bias, and the input should be constant 1.0
                            else:
                                trainingPointInput = 1.0
                            # Compute delta w and apply the change
                            inputNorm = 0
                            for tpi in trainingPointInputs:
                                inputNorm += tpi ** 2
                            inputNorm = math.sqrt(inputNorm)
                            deltaW = lr * (targetVal - outputVal) * trainingPointInput / inputNorm**2
                            self.weights[outputIndex, inputWeightIndex] += deltaW
            # Compute accuracy
            accuracy /= numTrainingPoints 
            # If verbose == True, print accuuracy for each training epoch
            if verbose:
                print('Epoch ' + str(e+1) + ' / ' + str(epochs) + ' Accuracy: ' + str(accuracy))
            # Return final accuracy
        return accuracy


# Test the network's learning capabilities
if __name__ == '__main__':
    """
        Test the network on the AND and XOR Boolean functions. Show training progress
        in graphs.
    """
    # Params
    EPOCHS = 20
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
    and_aln = AdalineNetwork(2, 1, randomize=True, bias=True)
    xor_aln = AdalineNetwork(2, 1, randomize=True, bias=True)

    # Create the suplots
    fig, axes = plt.subplots(2, NUM_GRAPHS)
    fig.suptitle('Adaline Network Training on AND and XOR Boolean Functions',
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
        xor_aln.train(xor_train_x, xor_train_y, epochs=NUM_EPOCHS_PER_GRAPH, verbose=False)
        and_aln.train(and_train_x, and_train_y, epochs=NUM_EPOCHS_PER_GRAPH, verbose=False)

        # Plot the model
        plotModel(xor_aln, axes[0, i], [-1.2, 1.2, -1.2, 1.2], [-1.0, 1.0])
        axes[0, i].set_xlim(-1.2, 1.2)
        axes[0, i].set_ylim(-1.2, 1.2)
        axes[0, i].set_aspect('equal', adjustable='box')

        plotModel(and_aln, axes[1, i], [-1.2, 1.2, -1.2, 1.2], [-1.0, 1.0])
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
        Let's show all of them and how Adaline networks perform when using the
        Widrow-Hoff Learning Rule.
    """
    # Params
    EPOCHS_PER_MODEL = 50

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
                'Adaline Networks using Widrow-Hoff Learning Rule.',
                fontsize=14)

    # Print out a warning to user that this may take a while, but progress will be output below
    print('\n\nWARNING: This may take a while. Progress will be output below:\n\n')

    # Create models for each of the functions, and plot the trained models
    f, x, y = 0, 0, 0
    for function in functions:
        # Create Single Layer Network
        model = AdalineNetwork(2, 1, randomize=True, bias=True)

        # Train the model on its dataset
        training_x, training_y = function.getTrainingData()
        model.train(training_x, training_y, epochs=EPOCHS_PER_MODEL, verbose=False)

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

    # Set caption
    textstr = str('Notice how the Adaline networks learn the separating boundary\n' + 
                  'that minimizes the MSE for the dataset (except XOR and XNOR which\n' +
                  'are non-linear). The network converges on the best unique value\n' +
                  'to classify these functions.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    plt.text(-15, 2.5, textstr, fontsize=14, bbox=props)

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


    """
        Adaline networks work with continuous inputs as well. Let's create
        a random clustered dataset and see how well the Adaline network classifies this data.
    """
    # Params
    EPOCHS = 20
    NUM_GRAPHS = 5
    
    NUM_POINTS_PER_CLASS = 40
    NOISE = 4

    NUM_EPOCHS_PER_GRAPH = math.ceil(EPOCHS / NUM_GRAPHS)
    
    # Create the training data
    train_x = []
    train_y = []

    classOneLine = (1, -2)
    classTwoLine = (5, 3)

    xvalues = []
    classOneY = []
    classTwoY = []

    for _ in range(NUM_POINTS_PER_CLASS):
        x = random.random() * 10
        y1 = classOneLine[0] * x + classOneLine[1] + (random.random()-0.5) * 2.0 * NOISE
        y2 = classTwoLine[0] * x + classTwoLine[1] + (random.random()-0.5) * 2.0 * NOISE
        
        xvalues.append(x)
        classOneY.append(y1)
        classTwoY.append(y2)

        train_x.append([x, y1])
        train_y.append([-1])

        train_x.append([x, y2])
        train_y.append([1])

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    # Create the network
    adaline = AdalineNetwork(2, 1, randomize=False, bias=True)

    # Create the subplots
    fig, axes = plt.subplots(1, NUM_GRAPHS)
    fig.suptitle('Adaline Network training on randomized dataset\n' +
                 'with continuous values, and a learnable (but not completely\n' +
                 'linearly separable) rule.',
                 fontsize=14)

    # Label the subplots
    for y in range(1, NUM_GRAPHS+1):
        axes[y-1].set_title('Adaline Network at\n' + str(y * int(NUM_EPOCHS_PER_GRAPH)) + ' Epochs')
    
    # Train each model on their respective datasets and show progress in the form
    # of decision boundaries for every N epochs
    for i in range(NUM_GRAPHS):
        # Train the model for N more epochs
        adaline.train(train_x, train_y, epochs=NUM_EPOCHS_PER_GRAPH, verbose=False)

        # Plot the model
        plotModel(adaline, axes[i], [-10.2, 20.0, -10.2, 20.0], [-1.0, 1.0], steppingSize=0.1)
        axes[i].set_xlim(-10.2, 20.0)
        axes[i].set_ylim(-10.2, 20.0)
        axes[i].set_aspect('equal', adjustable='box')

        # Plot the actual data for the model
        axes[i].scatter(xvalues, classOneY, c='red', s=50)
        axes[i].scatter(xvalues, classTwoY, c='blue', s=50)

    # Plot the graph
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    