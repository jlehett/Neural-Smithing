import numpy as np
from math import e

"""
    Functions
"""

# Sigmoid function for activation
@np.vectorize
def sigmoid(x):
    return 1.0 / (1.0 + e**(-x))

# Defined derivative of sigmoid
@np.vectorize
def sigmoidDerivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

# Define the SSE error function
def SSE(inputs, network, targetOutputs, activationFunction=sigmoid):
    # Convert outputs to numpy array for easier calculations
    targetOutputs = np.asarray(targetOutputs)
    # First, compute the network outputs on the given inputs
    networkOutputs, _ = feedforward(inputs, network, sigmoid)
    # Compute the SSE for the network on the given inputs
    
    # Compute the squared difference of each output to its corresponding
    # targets. This is done easily using numpy arrays.
    squaredDifferenceArray = (networkOutputs - targetOutputs) ** 2
    # Add each of these errors up. Again this is done easily in numpy.
    totalError = np.sum(squaredDifferenceArray)
    # Divide the total error by 2 and you have the final resulting SSE.
    totalError /= 2.0
    return totalError

# Feedforward function for determining output of network given an input
def feedforward(inputs, network, activationFunction):
    # For keeping records
    layerOutputs = []
    # convert inputs to numpy array for easier calculations
    inputs = np.asarray(inputs)
    # Iterate through each layer of the network
    currentLayer = inputs
    for layerNum in range(len(network.weights)):
        # Perform weighted addition
        currentLayer = np.dot(currentLayer, network.weights[layerNum])
        # Apply activation function
        currentLayer = activationFunction(currentLayer)
        # Record layer output
        layerOutputs.append(currentLayer)
    # Return the final output and the layer records
    return currentLayer, layerOutputs


"""
    FeedforwardNetwork class that stores the weights necessary for the
    network to operate.
"""
class FeedforwardNetwork:

    def __init__(
        self, inputSize, hiddenSizes, outputSize, randomize=False
        ):
        """
            Create a basic multi-layer feedforward network.
        """
        # Set the input size
        self.inputSize = inputSize
        # Set the hidden layer weights
        self.weights = []
        for i in range(len(hiddenSizes)):
            # Get previous layer size
            if i == 0:
                prevLayerSize = self.inputSize
            else:
                prevLayerSize = hiddenSizes[i-1]
            # Set the weights for the current layer
            if not randomize:
                self.weights.append(
                    np.zeros((prevLayerSize, hiddenSizes[i]))
                )
            if randomize:
                self.weights.append(
                    np.random.rand(prevLayerSize, hiddenSizes[i]) * 2.0 - 1.0
                )
        # Set the weights for the final layer
        if not randomize:
            self.weights.append(
                np.zeros((hiddenSizes[-1], outputSize))
            )
        if randomize:
            self.weights.append(
                np.random.rand(hiddenSizes[-1], outputSize) * 2.0 - 1.0
            )