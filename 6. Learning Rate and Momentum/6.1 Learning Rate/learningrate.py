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


"""
    A 4/4/1 network was trained to learn the 4-bit parity problem
    using plain batch back-propagation. Input values were +-1 and target
    values were +-0.9. Tanh nonlinearities were used at the hidden and
    output nodes. Initial weights were uniformly distributed in 
    [-0.5, +0.5]. A single nonadaptive learning rate was used for the 
    entire network.

    Mean-Squared-Error was used as the error function. A variety of
    learning rates from 0.001 to 10 (37 values) and momentums rom 0
    to 0.99 (14 values) were tested. One hundred trials were run for
    each parameter pair. Each network was allowed 5000 training epochs.
    Learning was considered successful (converged) if MSE < 0.001 or
    if every pattern was classified correctly with error less than 0.2.

    If the change in MSE between epochs was less than 10^(-12) or if the
    magnitude of the gradient was less than 10^(-10), the network was 
    considered stuck and training was halted early. The average
    convergence time for each parameter pair was calculated as the sum
    of the times for the converged networks divided by the number of
    networks that converged. If no networks converged for a particular
    set of parameters, avg time was set to 5000 for graphing purposes.
    The probability of convergence was estimated as the number of networks
    that converged divided by the number trained (100).
"""

# Define the 4-bit parity problem
def createAllBit4Strings():
    x, y = [], []
    maxValue = 4**2 - 1
    rangeValues = range(maxValue)
    for value in rangeValues:
        evenOnes = True
        binaryString = '{0:04b}'.format(value)
        binaryStringArray = list(binaryString)
        for i in range(len(binaryStringArray)):
            binaryStringArray[i] = int(binaryStringArray[i])
            if binaryStringArray[i] == 1:
                evenOnes = not evenOnes
        x.append(binaryStringArray)
        if evenOnes:
            y.append(1)
        else:
            y.append(0)
    return x, y

x, y = createAllBit4Strings()

