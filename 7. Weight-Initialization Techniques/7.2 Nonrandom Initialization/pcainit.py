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
from sklearn.decomposition import PCA
import torch.optim as optim


"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.auxfunctions import mapRange


"""
    Principal components analysis (PCA) attempts to identify the 
    major axes of variation of a data set -- the directions along 
    which the data varies the most. 
    
    Going on the assumption that these components are important 
    directions for the function to be learned, one can initialize 
    the input-to-hidden weight vectors along these directions. In a
    network with H hidden nodes the H eigenvectors with the largest
    eigenvalues would be selected.
    
    In the first phase of training, the input-to-hidden weights
    are fixed to the principal component directions while the 
    output weights are trained. In the second phase, all weights 
    are allowed to learn.
"""

# Set the pytorch device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# Create the Neural Network class via PyTorch
class PCANetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, lr=0.1):
        super().__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        # Define network layers
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, outputSize)

        self.to(device)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.0)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Define forward pass
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def initPCA(self, x):
        # Perform PCA on the dataset
        pca = PCA(n_components=self.hiddenSize)
        pca.fit_transform(x)
        # Initialize the first set of weights based on the PCA eigenvectors
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.from_numpy(pca.components_).float())

    def trainNetwork(self, dataloader, epochs):
        self.train()
        # Phase 1
        self.fc1.weight.requires_grad = False
        losses = []
        for epoch in range(round(epochs/2)):

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets = targets.view(-1, 1)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            losses.append(running_loss)
        
        # Phase 2
        self.fc1.weight.requires_grad = True
        for epoch in range(round(epochs/2)):

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets = targets.view(-1, 1)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            losses.append(running_loss)
        return np.asarray(losses)