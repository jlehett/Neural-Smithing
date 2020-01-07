## 5.3 Back-Propagation: The Weight Update Algorithm

Having obtained the derivatives, the next step is to update the weights so as to decrease the error. Back-propagation is
basically equivalent to gradient descent. By definition, the gradient of <i>E</i> points in the direction that increases <i>
E</i> the fastest. In order to minimize <i>E</i>, the weights are adjusted in the opposite direction. The weight update
formula is

<i>Δ w<sub>ij</sub></i> = - <i>η</i> <i><sup>∂ E</sup></i> / <i><sub>∂ w<sub>ij</sub></sub></i>

where the <i>learning rate η</i> > 0 is a small positive constant. If the derivative is positive (so increases in <i>w</i>
causes increases in <i>E</i>) then the weight change is negative and vice versa. This approaches pure gradient descent when
<i>η</i> is infinitesimal. Typical values for learning rate are in the range 0.05 < <i>η</i> < 0.75.

The network is usually initialized with small random weights. Values are often selected uniformly from a range [-<i>a</i>, 
+<i>a</i>] where 0.1 < <i>a</i> < 2 typically. Random values are needed to break symmetry while small values are necessary
to avoid immediate saturation of the sigmoid nonlinearities.

There are two basic weight-update variations, <i>batch-mode</i> and <i>on-line</i>.

### Batch Learning

In batch-mode, every pattern <i>p</i> is evaluated to obtain the derivative terms <i>∂ E<sub>p</sub></i> / <i>∂ w</i>; these
are summed to obtain the total derivative

<i><sup>∂ E</sup></i> / <i><sub>∂ w</sub></i> = <b>SUM</b> <sub><i>p</i></sub> [ <i><sup>∂ E<sub>p</sub></sup></i> / 
<i><sub>∂ w</sub></i> ]

and only then are the weights updated. Each pass through the training set is called an <i>epoch</i>. Batch-mode learning
approximates gradient descent when the learning rate <i>η</i> is small. In general, each weight update reduces the error
by only a small amount so many epochs are needed to minimize the error.

The code in this section titled "batch.py" builds upon the neural network class being constructed in sections 5.1 and 5.2,
and demonstrates how to build the batch learning capabilities. The network then trains on a simple boolean function dataset
for a specified number of epochs, and outputs the following results:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/5.%20Back-Propagation/5.3%20Back-Propagation:%20The%20Weight%20Update%20Algorithm/images/1.png)

### On-Line Learning

In <i>on-line</i> or <i>pattern-mode</i> learning, the weights are updated after each pattern presentation. The output is 
compared with the target for that pattern and the errors are back-propagated to obtain the single-pattern derivative
<i>∂ E<sub>p</sub></i> / <i>∂ w</i>. The weights are then updated immediately, using the gradient of the single-pattern 
error. Generally, the patterns are presented in a random, constantly changing order to avoid cyclic effects.

An advantage of this approach is that there is no need to store and sum the individual <i>∂ E<sub>p</sub></i> / <i>∂ w</i> 
contributions; each pattern derivative is evaluated, used immediately, and then discarded. Another possible advantage is
that many more weight updates occur in a given amount of time. If the training set contains <i>M</i> patterns, for example,
on-line learning would make <i>M</i> weight changes in the time that batch-mode learning makes only one.

The code in this section titled "online.py" builds upon the neural network class being constructed in sections 5.1 and 5.2,
and demonstrates how to build the on-line learning capabilities. The network then trains on a simple boolean function 
dataset for a specified number of epochs, and outputs the following results:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/5.%20Back-Propagation/5.3%20Back-Propagation:%20The%20Weight%20Update%20Algorithm/images/2.png)

### Comparison

In on-line learning, instead of following a smooth trajectory down the gradient, the weight vector tends to jitter around
the <i>E</i>(<i>w</i>) surface, mostly moving downhill, but occasionally jumping uphill. When pure gradient descent, or
batch-mode learning, arrives at a local minimum, it can get stuck. In on-line learning, the weight state tends to jitter
around its equilibrium value. This may bounce the weight vector out of a poor minimum and find a better solution.

However, because of this jittering, the weight vector in on-line learning never settles to a stable value. Having found
a good minimum, it may then wander off.

### [Continue to Section 5.4](https://github.com/jlehett/Neural-Smithing/tree/master/5.%20Back-Propagation/5.4%20Common%20Modifications)
