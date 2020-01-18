"""
    EXTERNAL LIBRARIES
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import keras


"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import mapRange


"""
    A demonstration of the effect of various learning rates and momentum
    values on the time to converge for training neural networks.

    Network was trained on the 2-bit parity problem. Input values were 
    +- 1 and target values were +- 0.9. Tanh nonlinearities were used
    at hidden and output nodes. Initial weights were uniformly distributed
    in [-0.5, +0.5]. A single nonadaptive learning rate was used for the
    entire network.

    A variety of learning rates from 0.001 to 5.0 (50 values) and 
    momentums from 0.0 to 0.99 (14 values) were tested. 10 trials were
    run for each parameter pair, and each network was allowed 750
    training epochs. Learning was considerd successful (converged) if
    MSE < 0.001 or if every training pattern was classified correctly
    with error less than 0.2.

    The average convergence time for each parameter pair was calculated
    as the sum of the times for the converged networks divided by the
    number of networks that converged. The probability of convergence
    was estimated as the number of networks that converged divided by
    the number trained (10).
"""

# Define the 2-bit parity problem
def createAllBit2Strings():
    x, y = [], []
    maxValue = 2**2 - 1
    rangeValues = range(maxValue)
    for value in rangeValues:
        evenOnes = True
        binaryString = '{0:02b}'.format(value)
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

x, y = createAllBit2Strings()

# Define the neural network
def createNetwork(lr, momentum):
    weightInitializer = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)

    network = keras.models.Sequential()
    network.add(keras.layers.Dense(
        4, activation='tanh', kernel_initializer=weightInitializer,
        bias_initializer=weightInitializer
    ))
    network.add(keras.layers.Dense(
        1, activation='tanh', kernel_initializer=weightInitializer,
        bias_initializer=weightInitializer
    ))

    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    network.compile(
        loss='mean_squared_error', optimizer=optimizer,
        metrics=['mean_squared_error']
    )

    return network

# Create function to train networks, halting if the network is considered
# stalled, as defined above
def trainNetwork(network, epochs):
    prevMSE = None
    for e in range(epochs):
        history = network.fit(x, y, verbose=0)
        newMSE = history.history['mean_squared_error'][0]
        if networkConverged(network, history):
            return e, True
    return epochs, False
        
# Create a function that determines whether a network has converged or not
def networkConverged(network, history):
    if history.history['mean_squared_error'][0] < 0.001:
        return True
    predictions = network.predict(x)
    for y_index in range(len(predictions)):
        pred_y = predictions[y_index]
        true_y = y[y_index]
        if (pred_y - true_y) ** 2 >= 0.2:
            return False
    return True

# Create a function to obtain results for a parameter pair
def getParameterPairResults(lr, momentum):
    maxEpochs = 750
    numConverged = 0
    timeToConverge = 0
    numTrials = 10
    
    for i in range(numTrials):
        network = createNetwork(lr, momentum)
        numEpochsToConverge, converged = trainNetwork(network, maxEpochs)
        if converged:
            timeToConverge += numEpochsToConverge
            numConverged += 1
    
    if numConverged == 0:
        avgTimeToConverge = maxEpochs
    else:
        avgTimeToConverge = timeToConverge / numConverged
    probabilityConvergence = numConverged / numTrials
    return avgTimeToConverge, probabilityConvergence

# Get the range of learning rates to test
learningRates = []
for i in range(50):
    learningRates.append(mapRange(i, 0, 49, 0.001, 5.0))

# Get the range of momentums to test
momentums = []
for i in range(14):
    momentums.append(mapRange(i, 0, 13, 0.0, 0.99))

# Create function to plot a momentum value
def plotMomentumValue(momentum):
    # Obtain the data
    data_x = learningRates
    data_avgTime = []
    data_probability = []
    
    progress = 1
    for lr in learningRates:
        avgTime, probability = getParameterPairResults(lr, momentum)
        data_avgTime.append(avgTime)
        data_probability.append(probability)
        progress += 1
    # Plot the data
    fig, ax1 = plt.subplots()
    ax1.title.set_text('Momentum Value = {:.2f}'.format(momentum))

    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Avg Epochs to Convergence')
    ax1.plot(data_x, data_avgTime)
    
    ax2 = ax1.twinx()

    ax2.set_ylabel('Probability of Convergence')
    ax2.plot(data_x, data_probability, linestyle='--')

    fig.tight_layout()
    ax1.set_xticks(np.arange(0, 5.1, step=0.5))
    ax1.set_yticks(np.arange(0, 750.1, step=75))
    ax2.set_yticks(np.arange(0, 1.1, step=0.2))
    plt.show()


for momentum in momentums:
    plotMomentumValue(momentum)
