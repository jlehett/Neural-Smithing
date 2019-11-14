"""
    EXTERNAL LIBRARIES
"""

import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Set environment log levels for training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import plot_decision_boundary, getWeights, plotHyperplane


"""
    The exclusive-OR function is a well-known example of a simple
    function that is not linearly separable and thus not computable
    by single-layer perceptrons.
"""
# Params
TOTAL_EPOCHS = 10000
NUM_GRAPHS_PER_MODEL = 5

# Create a single-layer model via Keras (to ensure model functions properly)
xor_model = keras.models.Sequential()
xor_model.add(keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)))
xor_model.compile(optimizer='adam', loss='mean_squared_error')

# Create the XOR dataset and fit the xor_model to it
xor_x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
xor_y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Create a single-layer model via keras (to ensure model functions properly)
and_model = keras.models.Sequential()
and_model.add(keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)))
and_model.compile(optimizer='adam', loss='mean_squared_error')

and_x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
and_y = np.array([
    [0],
    [0],
    [0],
    [1]
])

# Create the pyplot
fig, axes = plt.subplots(2, NUM_GRAPHS_PER_MODEL)
fig.suptitle('XOR Model cannot be fit with a single-layer network. ' +
             'AND Model can.\n These graphs display decision boundaries.',
             fontsize=18)

# Label the subplots
for x in range(1, 3):
    for y in range(1, NUM_GRAPHS_PER_MODEL+1):
        if x == 1:
            axes[x-1][y-1].set_title('XOR Model at\n' + str(y * int(TOTAL_EPOCHS / NUM_GRAPHS_PER_MODEL)) + ' Epochs')
        else:
            axes[x-1][y-1].set_title('AND Model at\n' + str(y * int(TOTAL_EPOCHS / NUM_GRAPHS_PER_MODEL)) + ' Epochs')

# Print out a warning to user that this may take a while, but progress will be output below
print('\n\nWARNING: This may take a while. Progress will be output below:\n\n')

# Train each model on their respective datasets and show progress in the form
# of decision boundaries for every 1000 epochs
for i in range(NUM_GRAPHS_PER_MODEL):
    # Train each model for 1000 more epochs
    xor_model.fit(xor_x, xor_y, epochs=int(TOTAL_EPOCHS / NUM_GRAPHS_PER_MODEL), verbose=0)
    and_model.fit(and_x, and_y, epochs=int(TOTAL_EPOCHS / NUM_GRAPHS_PER_MODEL), verbose=0)

    # Print progress
    print('PROGRESS: ' + str(i+1) + ' / ' + str(NUM_GRAPHS_PER_MODEL))

    # Plot the new decision boundaries for each model
    plot_decision_boundary(lambda x: xor_model.predict([x]), [0, 1, 0, 1], axes[0, i])
    plot_decision_boundary(lambda x: and_model.predict([x]), [0, 1, 0, 1], axes[1, i])

    # Plot the actual data for each model
    axes[0, i].scatter([0, 1], [1, 0], c='blue', s=150)
    axes[0, i].scatter([0, 1], [0, 1], c='red', s=150)

    axes[1, i].scatter([1], [1], c='blue', s=300)
    axes[1, i].scatter([0, 0, 1], [0, 1, 0], c='red', s=150)

    # Plot the hyperplane for each model
    plotHyperplane(xor_model, axes[0, i], 10, 'Hyperplane')
    plotHyperplane(and_model, axes[1, i], 10, 'Hyperplane')

    # Adjust graph parameters
    axes[0, i].set_xlim(-0.2, 1.2)
    axes[0, i].set_ylim(-0.2, 1.2)
    axes[0, i].set_aspect('equal', adjustable='box')

    axes[1, i].set_xlim(-0.2, 1.2)
    axes[1, i].set_ylim(-0.2, 1.2)
    axes[1, i].set_aspect('equal', adjustable='box')

# Plot the graph
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()


"""
    14 of the 16 Boolean functions are linearly separable.
    Let's show all of them and how Single-Layer Networks in keras perform
    on them.
"""
# Params
EPOCHS_PER_MODEL = 5000

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
            training_y.append(1)
        for pair in self.false_x:
            training_x.append(pair)
            training_y.append(0)
        return np.array(training_x), np.array(training_y)

# Define function to plot a trained model's results
def plotModel(model, true_x, false_x, ax, title):
    # Plot the decision boundaries for the model
    plot_decision_boundary(lambda _x: model.predict([_x]), [0, 1, 0, 1], ax)

    # Plot the actual data for each model
    for pair in true_x:
        ax.scatter(pair[0], pair[1], c='blue', s=50)
    for pair in false_x:
        ax.scatter(pair[0], pair[1], c='red', s=50)
    
    # Plot the hyperplane for each model
    plotHyperplane(model, ax, 10, 'Hyperplane')

    # Adjust graph parameters
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    # Set the title of the subplot
    ax.set_title(title)

# Define each of the 16 boolean functions
functions = []
functions.append(
    BooleanFunction(
        [],
        [[0, 0], [0, 1], [1, 0], [1,1]],
        'FALSE'
    )
)
functions.append(
    BooleanFunction(
        [[1, 1]],
        [[0, 0], [0, 1], [1, 0]],
        'AND'
    )
)
functions.append(
    BooleanFunction(
        [[1, 0]],
        [[0, 0], [0, 1], [1, 1]],
        'A AND NOT B'
    )
)
functions.append(
    BooleanFunction(
        [[1, 0], [1, 1]],
        [[0, 0], [0, 1]],
        'A'
    )
)
functions.append(
    BooleanFunction(
        [[0, 1]],
        [[0, 0], [1, 0], [1, 1]],
        'NOT A AND B'
    )
)
functions.append(
    BooleanFunction(
        [[0, 1], [1, 1]],
        [[0, 0], [1, 0]],
        'B'
    )
)
functions.append(
    BooleanFunction(
        [[0, 1], [1, 0]],
        [[0, 0], [1, 1]],
        'XOR'
    )
)
functions.append(
    BooleanFunction(
        [[0, 1], [1, 0], [1, 1]],
        [[0, 0]],
        'OR'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0]],
        [[0, 1], [1, 0], [1, 1]],
        'NOR'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0], [1, 1]],
        [[0, 1], [1, 0]],
        'XNOR'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0], [1, 0]],
        [[0, 1], [1, 1]],
        'NOT B'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0], [1, 0], [1, 1]],
        [[0, 1]],
        'A OR NOT B'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0], [0, 1]],
        [[1, 0], [1, 1]],
        'NOT A'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0], [0, 1], [1, 1]],
        [[1, 0]],
        'NOT A OR B'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0], [0, 1], [1, 0]],
        [[1, 1]],
        'NAND'
    )
)
functions.append(
    BooleanFunction(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [],
        'TRUE'
    )
)

# Create the pyplot
fig, axes = plt.subplots(2, 8)
fig.suptitle('14 of 16 Boolean functions are linearly separable\n' +
             'These graphs display decision boundaries.',
             fontsize=14)

# Print out a warning to user that this may take a while, but progress will be output below
print('\n\nWARNING: This may take a while. Progress will be output below:\n\n')

# Create models for each of the functions, and plot the trained models
f, x, y = 0, 0, 0
for function in functions:
    # Create Single Layer Network
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on its dataset
    training_x, training_y = function.getTrainingData()
    model.fit(training_x, training_y, epochs=EPOCHS_PER_MODEL, verbose=0)

    # Print progress
    print('PROGRESS: ' + str(f+1) + ' / 16')

    # Plot the trained model in the appropriate subplot
    plotModel(model, function.true_x, function.false_x, axes[x, y], function.name)

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
