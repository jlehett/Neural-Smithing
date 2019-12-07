import numpy as np

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