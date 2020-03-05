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
    The number of training samples should be larger than the number
    of parameters divided by the desired approximation error,
        N > O(Mp / Ɛ)
    Here, N is the number of samples, M is the number of hidden nodes,
    p is the input dimension (so Mp is approximately the number of
    parameters), and Ɛ is the desired approximation error.
""" 

def getMinNumTrainingSamples(numHiddenNodes, inputDimension, desiredApproxError):
    """
        Function that defines the minimum number of training samples that
        should be used to achieved the desired approximation error
    """
    return numHiddenNodes * inputDimension / desiredApproxError

def getMinNumHiddenNodes(numTrainingSamples, inputDimension, desiredApproxError):
    """
        Function that defines the minimum number of hidden nodes that
        should be used to achieve the desired approximation error
    """
    return numTrainingSamples * desiredApproxError / inputDimension


# Get user input and find min number of training samples
print('\n\nFirst, we will try to find the minimum number of training ' +
      'samples that should be used to get the desired approximation error.')

numHiddenNodes = float(input('\nHow many hidden nodes are in the network? '))
inputDimension = float(input('How many dimensions is the input to the network? '))
desiredApproxError = float(input('What is the desired approximation error? '))

print('\nThe minimum number of training samples should be ' + 
      str(getMinNumTrainingSamples(numHiddenNodes, inputDimension, desiredApproxError)) +
      '.')

# Get user input and find min number of hidden nodes
print('\n\nNext, we will try to find the minimum number of hidden ' +
      'nodes that should be used to get the desired approximation error.')

numTrainingSamples = float(input('\nHow many training samples are available? '))
inputDimension = float(input('How many dimensions is the input to the network? '))
desiredApproxError = float(input('What is the desired approximation error? '))

print('\nThe minimum number of hidden nodes should be ' +
      str(getMinNumHiddenNodes(numTrainingSamples, inputDimension, desiredApproxError)) +
      '.')