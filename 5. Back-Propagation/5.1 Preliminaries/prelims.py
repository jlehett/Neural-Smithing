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
from utils.auxfunctions import FeedforwardNetwork


"""
    In the forward pass, the network computes an output based on its current inputs.
    Each node, i, computes a weighted sum Ai of its inputs an passes this through a
    nonlinearity to obtain the node output Yi.

    This process is detailed below. The FeedforwardNetwork class only contains an init
    function that stores the weights of each layer in an array. The code written below
    performs a forward pass on the network.
"""