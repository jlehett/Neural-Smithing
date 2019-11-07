"""
    EXTERNAL LIBRARIES
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from math import sqrt


plt.style.use('dark_background')

"""
    INTERNAL UTIL LIBRARIES
"""

# Add parent directory to the sys path
sys.path.append('../')
# Import any python modules from parent directory
from utils.SingleLayerNetwork import SingleLayerNetwork


"""
    The locus of points x with a constant sum
        SUM( Wj * Xj )
    defines a hyperplane perpendicular to the vector w.

    First, 2D visualization
"""
# Params
TARGET_MAG = 5      # The value to scale the weights by for plotting

# Define an activation function that does not change the input value.
# No activation function is needed for this visualization.
def noActivation(x):
    return x

for i in range(1):
    slp = SingleLayerNetwork(2, 1, noActivation, randomize=True)

    # Scale the W vector such that it has a magnitude of TARGET_MAG
    # and it is centered at (0, 0)
    slpWeights = slp.getWeights()[0]

    scaledWeightX = [
        -sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[0],
        sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[0]
    ]
    scaledWeightY = [
        -sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[1],
        sqrt(TARGET_MAG**2 / (slpWeights[0]**2 + slpWeights[1]**2)) * slpWeights[1]
    ]

    plt.arrow(scaledWeightX[0], scaledWeightY[0], 
            scaledWeightX[1]*2, scaledWeightY[1]*2, 
            head_width=0.5, overhang=0.2, linestyle='-', color='blue',
            linewidth=5)

    # Grab a point with the constant sum of 0
    if slpWeights[1] != 0:
        xcoord = 1
        ycoord = (0 - slpWeights[0] * xcoord) / slpWeights[1]
    else:
        ycoord = 1
        xcoord = (0 - slpWeights[1] * ycoord) / slpWeights[0]

    # Form a line given this point, scaled to TARGET_MAG
    scaledPointX = [
        -sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * xcoord,
        sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * xcoord
    ]
    scaledPointY = [
        -sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * ycoord,
        sqrt(TARGET_MAG**2 / (xcoord**2 + ycoord**2)) * ycoord
    ]

    for i in range(len(scaledPointX)):
        x = scaledPointX[i]
        y = scaledPointY[i]
        print(slp.evaluate(np.array([x, y])))

    # Plot the locus of points with constant sum of 0
    plt.arrow(scaledPointX[0], scaledPointY[0], 
            scaledPointX[1]*2, scaledPointY[1]*2,
            head_width=0.5, overhang=0.2, linestyle='-', color='red',
            linewidth=3)

    # Limit the x and y axis such that you get a square plot (no squashing)
    # Squashing would cause the line to look as though it weren't perpendicular.
    plt.xlim(-TARGET_MAG*1.3, TARGET_MAG*1.3)
    plt.ylim(-TARGET_MAG*1.3, TARGET_MAG*1.3)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()