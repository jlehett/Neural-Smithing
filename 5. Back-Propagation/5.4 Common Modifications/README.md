## 5.4 Common Modifications

### Momentum

A common modification of the basic weight update rule is the addition of a <i>momentum</i> term. The idea is to stabilize the
weight trajectory by making the weight change a combination of the gradient-decreasing term plus a fraction of the previous
weight change. The modified weight change formula is

Δ<b>w</b>(<i>t</i>) = - <i>η</i> <i><sup>∂ E</sup></i> / <sub><i>∂</i> <b>w</b></sub> (<i>t</i>) + <i>α</i> Δ<b>w</b>(<i>t
</i> - 1)

That is, the weight change Δ<b>w</b>(<i>t</i>) is a combination of a step down the negative gradient, -
<i>η</i> <i><sup>∂ E</sup></i> / <sub><i>∂</i> <b>w</b></sub> (<i>t</i>), plus a fraction 0 ≤ <i>α</i> < 1 of the previous
weight change. Typical values are 0 ≤ <i>α</i> < 0.9.

This gives the system a certain amount of inertia since the weight vector will tend to continue moving in the same direction
unless opposed by the gradient term. Momentum tends to damp oscillations in the weight trajectory and accelerate learning in
regions where <i>∂ E</i> / <i>∂ w</i> is small.

The code in this section titled "momentum.py" has a demonstration of how to add momentum to the training process in the
neural network class that was built in sections 5.1, 5.2, and 5.3. If you run the code, you will notice the error value
initially decreases at a very slow speed, but it picks up "momentum" until it stabilizes at a final value after being
opposed by the gradient. The final output of the code can be seen below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/5.%20Back-Propagation/5.4%20Common%20Modifications/images/1.png)

### Weight Decay

Another common modification of the weight update rule is the addition of a <i>weight decay</i> term. Weight decay is sometimes
used to help adjust the complexity of the network to the difficulty of the problem. The idea is that if the network is overly
complex, then it should be possible to delete many weights without increasing the error significantly. One way to do this is
to give the weights a tendency to drift to zero by reducing their magnitudes slightly at each iteration. The update rule with
weight decay is then

Δ<b>w</b>(<i>t</i>) = - <i>η</i> <i><sup>∂ E</sup></i> / <sub><i>∂</i> <b>w</b></sub> (<i>t</i>) - <i>ρ</i><b>w</b>(<i>t</i>)

where 0 ≤ <i>ρ</i> ≪ 1 is the weight decay parameter. If <i><sup>∂ E</sup></i> / <i><sub>∂ w<sub>i</sub></sub></i> = 0
for some weight <i>w<sub>i</sub></i>, then <i>w<sub>i</sub></i> will decay to zero exponentially. Otherwise, if the weight
really is necessary, then <i><sup>∂ E</sup></i> / <i><sub>∂ w<sub>i</sub></sub></i> will be nonzero and the two terms will
balance at some point, preventing the weight from decaying to zero.

The code in this section titled "decay.py" has a demonstration of how to add weight decay to the training process in the
neural network class that was built in sections 5.1, 5.2, and 5.3. In the final output of the code, the number of 
"irrelevant" weights will be shown. This is determined by counting the number of weights that have decayed to a small range
around 0. In the final output, these weights are all set to 0.0, and the forward pass is then performed so you can see the
outputs of the network after removing these unnecessary weights. The output is shown below:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/5.%20Back-Propagation/5.4%20Common%20Modifications/images/2.png)

### [Continue to Section 6.1](https://github.com/jlehett/Neural-Smithing/tree/master/6.%20Learning%20Rate%20and%20Momentum/6.1%20Learning%20Rate)
