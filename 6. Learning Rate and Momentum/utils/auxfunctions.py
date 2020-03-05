"""
    Imported libraries
"""

import numpy as np
from math import e
import matplotlib.pyplot as plt


"""
    Aux Functions
"""

# Map a value from one range to another
def mapRange(value, min1, max1, min2, max2):
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1)