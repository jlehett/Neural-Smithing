"""
    EXTERNAL LIBRARIES
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import keras
import keras.backend as K


"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import mapRange


"""
    Demonstrates the E(t) curves for small SSE learning rates. At low
    learning rates, the E(t) curves are smooth, but convergence is slow.
    As learning rate increases, convergence time decreases but convergence
    is less reliable with occasional jumps in error.
"""

# Define the 4-bit parity problem
def createAllBit4Strings():
    x, y = [], []
    maxValue = 2**4 - 1
    rangeValues = range(maxValue)
    for value in rangeValues:
        evenOnes = True
        binaryString = '{0:04b}'.format(value)
        binaryStringArray = list(binaryString)
        for i in range(len(binaryStringArray)):
            binaryStringArray[i] = int(binaryStringArray[i])
            if binaryStringArray[i] == 0:
                binaryStringArray[i] = -1
            if binaryStringArray[i] == 1:
                evenOnes = not evenOnes
        x.append(binaryStringArray)
        if evenOnes:
            y.append(0.9)
        else:
            y.append(-0.9)
    return np.asarray(x), np.asarray(y)

x, y = createAllBit4Strings()

# Define the SSE error function
def SSE(y_actual, y_predicted):
    return K.square(y_predicted - y_actual)

# Define the neural network
def createNetwork(lr, randomSeed):
    weightInitializer = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=randomSeed)

    network = keras.models.Sequential()
    network.add(keras.layers.Dense(
        4, activation='tanh', kernel_initializer=weightInitializer,
        bias_initializer=weightInitializer
    ))
    network.add(keras.layers.Dense(
        1, activation='tanh', kernel_initializer=weightInitializer,
        bias_initializer=weightInitializer
    ))

    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.5)
    network.compile(
        optimizer=optimizer,
        loss=SSE,
        metrics=[SSE]
    )

    return network

# Create function to train networks, returning the training history
def trainNetwork(network, epochs):
    history = network.fit(x, y, verbose=0, epochs=epochs)
    return history

# Plot the error history for the given learning rate
def plotError(ax, errorHistory, label):
    ax.plot(errorHistory, label=label)

# Create the error curve graph given the learning rates to be tested
def createErrorGraph(learningRates, randomSeed):
    fig, ax = plt.subplots(1, 1)

    for learningRate in learningRates:
        network = createNetwork(learningRate, randomSeed)
        history = trainNetwork(network, 500)
        errorHistory = history.history['SSE']
        plotError(ax, errorHistory, 'η = ' + str(learningRate))
    
    ax.title.set_text('E(t) over Time\nα = 0.5')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('E(t)\nSSE')

    ax.legend()

    plt.show()


createErrorGraph([1.5, 1.25, 1.0, 0.5, 0.1], 500)
