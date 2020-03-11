"""
    EXTERNAL LIBRARIES
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils import data
from sklearn import datasets

"""
    INTERNAL UTIL LIBRARIES
"""

# Import any python modules from parent directory
from utils.auxfunctions import mapRange

# Import networks from sections 7.1 and 7.2
sys.path.append('./7.1 Random Initialization')
from randominit import OneHiddenNetwork as RandomNetwork

sys.path.append('./7.2 Nonrandom Initialization')
from pcainit import PCANetwork
from dainit import DANetwork


"""
    Make comparison charts for each different initialization
    scheme.
"""

# Set the pytorch device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Parameters
num_to_test = 10
epochs = 1000
lr = 0.1

# Store losses in a dictionary for each initialization scheme
lossDict = {
    'PyTorch Default': np.asarray([0 for _ in range(epochs)]),
    'Uniform Range A = 2': np.asarray([0 for _ in range(epochs)]),
    'Uniform Range A = 3': np.asarray([0 for _ in range(epochs)]),
    'Uniform Range No Variable': np.asarray([0 for _ in range(epochs)]),
    'Uniform Calculated': np.asarray([0 for _ in range(epochs)]),
    'Gaussian Calculated': np.asarray([0 for _ in range(epochs)])
}

# Test each scheme on num_to_test different random datasets for the number of
# epochs specified
for i in range(num_to_test):
    # Create the dataset for this trial
    x, y = datasets.make_blobs(
        n_samples=400, n_features=5, centers=4, cluster_std=0.1,
        center_box=(-1.0, 1.0), shuffle=True, random_state=None
    )

    # Create PyTorch dataloader from the generated dataset
    tensor_x = torch.from_numpy(x).float()
    tensor_y = torch.from_numpy(y).float()
    dataset = data.TensorDataset(tensor_x, tensor_y)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Train a network that is initialized with PyTorch defaults.
    network = RandomNetwork(5, 4, 1, lr)
    lossDict['PyTorch Default'] += network.trainNetwork(dataloader, epochs)

    # Train a network that is randomly initialized to a uniform range with A = 2
    network = RandomNetwork(5, 4, 1, lr)
    network.initRandom(2)
    lossDict['Uniform Range A = 2'] += network.trainNetwork(dataloader, epochs)

    # Train a network that is randomly initialized to a uniform range with A = 3
    network = RandomNetwork(5, 4, 1, lr)
    network.initRandom(3)
    lossDict['Uniform Range A = 3'] += network.trainNetwork(dataloader, epochs)

    # Train a network that is randomly initialized to uniform range (-2.4/N, 2.4/N)
    network = RandomNetwork(5, 4, 1, lr)
    network.initRandomUniformCalculated(
        max(
            max(np.amax(x[:, 0]), abs(np.amin(x[:, 0]))),
            max(np.amax(x[:, 1]), abs(np.amin(x[:, 1])))
        )
    )
    lossDict['Uniform Calculated'] += network.trainNetwork(dataloader, epochs)

    # Train a network that is randomly initialized to uniform range (-2.4/N, 2.4/N)
    network = RandomNetwork(5, 4, 1, lr)
    network.initGaussianCalculated(
        max(
            max(np.amax(x[:, 0]), abs(np.amin(x[:, 0]))),
            max(np.amax(x[:, 1]), abs(np.amin(x[:, 1])))
        )
    )
    lossDict['Gaussian Calculated'] += network.trainNetwork(dataloader, epochs)

    # Print progress
    print('Cycle ' + str(i+1) + ' / ' + str(num_to_test) + ' COMPLETE . . .')