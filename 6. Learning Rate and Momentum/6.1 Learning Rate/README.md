## 6.1 Learning Rate

The following example is used to illustrate how different choices of learning rate and momentum affect the training process.

Networks were trained on the 2-bit parity problem. Input values were ± 1 and target values were ± 0.9. Tanh nonlinearities
were used at hidden and output nodes. Initial weights were uniformly distributed in [-0.5, +0.5]. A single nonadaptive
learning rate was used for the entire network.

A variety of learning rates from 0.001 to 5.0 (50 values) and momentums from 0.0 to 0.99 (14 values) were tested. 10 trials
were run for each parameter pair, and each network was allowed 750 training epochs. Learning was considered successful
(converged) if E<sub>MSE</sub> < 0.001 or if every training pattern was classified correctly with error less than 0.2.

The average convergence time for each parameter pair was calculated as the sum of the times for the converged networks
divided by the number of networks that converged. The probability of convergence was estimated as the number of networks
that converged divided by the number trained (10).

Some of the results can be seen below. To save space, not all of the graphs will be included in this README. To see all of
the results, navigate to the "images" directory in this section. Probability of convergence is represented by the dotted line
and the y-axis on the right hand side ranging from 0.0-1.0. Convergence time (number of epochs to convergence) is represented
by the solid line and the y-axis on the left hand side ranging from 0-750.

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate/images/1.png)

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate/images/4.png)

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate/images/8.png)

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate/images/14.png)

In these graphs, we can see that as the learning rate increases, the convergence time begins to shrink until it reaches a 
critical point where any higher learning rates can cause the network to derail and never converge. 

As momentum values increase, the learning rate needed to achieve very quick results shrinks rapidly. However, at the same
time, the range of acceptable learning rates shrinks.

### <i>E</i>(<i>t</i>) Graphs

The following figures demonstrate the <i>E</i>(<i>t</i>) curves for networks trained with the same initial weight vector and momentum
value of 0.5, but with a range of learning rates. 

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate/images/15.png)

As this figure demonstrates, at low learning rates, the <i>E</i>(<i>t</i>) curves are smooth, but convergence is slow. As <i>η</i>
increases, convergence time decreases but convergence is less reliable with occasional jumps in error.

Once the critical learning rate point is surpassed, the system becomes unstable and may fail to converge at all. This change can 
sometimes be abrupt, and a network that converges at one rate may not for a rate that has increased as little as +0.01. The figure below shows the network becoming unstable for greater values of learning rates.

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate/images/16.png)

### Delta Attenuation in Layered Networks

The first layers of layered networks often learn very slowly because the error derivatives are attenuated as they propagate back
from the output layer toward the input. Each node nonlinearity contributes a derivative factor that is normally less than 1, so
the derivative may become very small after passing through several layers. The result is that the gradient tends to be very small
for weights close to the inputs and so they change very slowly.

Because the partial derivatives are so small, larger learning rates may be appropriate for hidden units. If no other information
is available, it might be assumed that the node outputs <i>y</i> are uniformly distributed on [0, 1] in which ase the expected
attenuation due to each sigmoid derivative is <sup>1</sup>/<sub>6</sub>. It can be suggested to re-scale the back-propagated
derivatives by 6 to compensate. This would be equivalent to increasing the learning rate by 6 for weights into the last layer,
by 36 for weights into the second-to-last-layer, 6<sup>3</sup> = 216 for weights 3 layers back from the output, and so on. These
are only heuristics, however, it is not a necessary fact that partial derivatives are smaller for weights farther from the
outputs.

An example of this heuristic being implemented in PyTorch on a deep neural network training on the 4-bit parity problem is
provided in the <b>deltaattenuation.py</b> file included in this section. It produces a graph showing the <i>E</i>(<i>t</i>)
curve over time for both a normal graph and a delta attenuated graph. Both networks were trained on the same 4-bit parity problem
for 10,000 epochs. The graph produced is given below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate/images/17.png)
