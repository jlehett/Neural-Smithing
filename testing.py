import numpy as np

deltas = np.asarray([1, 2])

activations = np.asarray([3, 4])

print(
    np.outer(activations, deltas)
)

real = []

real.append(
    [
        deltas[0] * activations[0],
        deltas[1] * activations[0]
    ]
)

real.append(
    [
        deltas[0] * activations[1],
        deltas[1] * activations[1]
    ]
)

print(real)