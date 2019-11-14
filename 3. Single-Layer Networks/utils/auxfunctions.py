import numpy as np
import matplotlib.pyplot as plt

def frange(start, stop, step):
    """
        Range function that accepts floats.
        Will produce the range of numbers from start to stop (inclusive)
        with step value provided. If stop is not produced exactly, will
        only produce the value right before it (not over).
    """
    i = start
    _range = []
    while i < stop:
        _range.append(i)
        i += step
    return _range

def plot_decision_boundary(pred_func, graphDims, ax):
    # Set min and max values and give it some padding
    x_min, x_max = graphDims[0] - .5, graphDims[1] + .5
    y_min, y_max = graphDims[2] - .5, graphDims[3] + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu)