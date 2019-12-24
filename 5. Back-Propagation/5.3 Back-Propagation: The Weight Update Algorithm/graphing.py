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
from utils.auxfunctions import sigmoid, sigmoidDerivative
from batch import BatchNetwork
from online import OnlineNetwork


"""
    Instead of smoothly rolling down the error gradient, like Batch
    training networks do, the weight vector dances along a semi-random
    path, mostly moving downhill, but occasionally jumping uphill in
    on-line learning networks.

    This script will help visualize the differences.
"""

# Create a dataset
@np.vectorize
def eq1(x):
    return x**3 + 20

@np.vectorize
def eq2(x):
    return x**3 - 20

plt.plot(
    np.arange(-5, 5, 0.1),
    eq1(np.arange(-5, 5, 0.1)),
    'bo'
)

plt.plot(
    np.arange(-5, 5, 0.1),
    eq2(np.arange(-5, 5, 0.1)),
    'go'
)
    
plt.show()