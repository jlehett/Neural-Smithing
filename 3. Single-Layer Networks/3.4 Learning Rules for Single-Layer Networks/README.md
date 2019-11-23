## 3.4 Learning Rules for Single-Layer Networks

### The Perceptron Learning Algorithm

The perceptron learning algorithm is suitable for learning linearly separable binary functions of binary inputs and is guaranteed to find a solution if the classes are linearly separable. If the classes are not linearly separable, the algorithm may not even converge and is unlikely to produce the best solution when it does.

Because the outputs are binary, linear threshold units are used. Each unit computes the weighted sum <i>u</i> of its <i>N</i> inputs <i>x<sub>j</sub></i>, <i>j</i> = 1 . . . <i>N</i>, and generates a binary output <i>y</i>.

<i>u</i> = SUM<sup><i>N</i></sup><sub><i>j</i> = 0</sub> [<i>w<sub>j</sub> x<sub>j</sub></i>] = <b>w<sup>T</sup>x</b>

<i>y</i> = {<sup>-1, <i>u</i> ≤ 0</sup> <sub>+1, <i>u</i> > 0</sub>

A node threshold is absorbed into the weight vector by assuming the presence of a constant value bias unit, <i>x<sub>bias</sub></i> = 1. Input, output, and target values are assumed to be ±1.

During training, input patterns <b>x</b> are presented and the outputs <i>y</i>(<b>x</b>) are compared to the targets <i>t</i>(<b>x</b>). Weights are adapted by

Δ<b>w</b> = {<sup>2<i> ŋ t</i><b> x</b>, if <i>t</i> ≠ <i>y</i></sup> <sub>0, otherwise</sub>

where <i>ŋ</i> is a small positive constant controlling the learning rate. Typically 0 < <i>ŋ</i> < 1.

This learning algorithm is first tested in detail on the AND and XOR datasets:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks/images/1.png)

As we can see above, the network <i>does</i> learn a decision boundary for AND, however it is only adjusted until it is 100% accurate. As such, as long as the datapoints are all classified correctly by the decision boundary, the network is unconcerned with creating better decision boundaries by separating the points further.

We test the learning algorithm next on all 16 Boolean functions:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks/images/2.png)

As expected, the network correctly learned the decision boundaries for all functions except XOR and XNOR.


### The Pocket Algorithm 

Although the perceptron algorithm is guaranteed to learn pattern sets that are linearly separable, it does not converge when the training data is not linearly separable. It is possible for the system to visit a near optimal set of weights and then wander away to a very bad set; the total error may be small at one moment and suddenly become large in the next so there is no guarantee that a weight vector obtained late in the training process will be near optimal.

Gallant's "pocket algorithm" keeps a copy of the best weight vector obtained so far. Two sets of weights are maintained, a working set and the pocket set. Whenever the current weights have a run of consecutive correct classifications longer than the best run by the pocket weighhts, they replace them. The quality of the pocket weights thus tends to imrpove over time and converge to an optimum set.

The pocket algorithm is first tested against the perceptron learning algorithm on the XOR dataset:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks/images/3.png)

As we can see, the pocket algorithm does not change its decision boundaries because it has not found a set of weights that gives a better classification score. The perceptron learning algorithm, on the other hand, is constantly changing.

A randomized dataset is then constructed. The perceptron learning algorithm and the pocket algorithm are both run on this dataset, with their classification scores being recorded each epoch and then plotted:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks/images/4.png)

As we can see, the pocket algorithm's score strictly increases and converges, whereas the perceptron learning algorithm is constantly fluctuating.

### Adalines and the Widrow-Hoff Learning Rule

The structure of an Adaline (<i>adaline linear neuron</i>) is a single-layer network with a step function as the nonlinearity. Adaline inputs may be continuous. Weights are adjusted with the Widrow-Hoff learning rule to minimize the difference between the output and an externally supplied target.

The Widrow-Hoff learning rule, also called the LMS algorithm or the delta rule, is basically an iterative implementation of linear regression. Both minimize the mean squared error of a linear fit.

An input pattern <b>x</b> is selected from the training set, applied to the input, and the output is calculated as in a normal single-layer network. The weights are then updated by:

Δ<b>w</b> = <i>ŋ</i>(<i>t</i> - <i>u</i>) * (<b>x</b> / ||<b>x</b>||<sup>2</sup>)

where <i>ŋ</i> is a small positive constant learning rate. Note that this minimizes the difference between the target and the weighted input sum <i>u</i>, not the output <i>y</i> = <i>f</i>(<i>u</i>).

The error is reduced by a factor of <i>ŋ</i> each time the weights are updated with the input pattern fixed. Stability requires that 0 < <i>ŋ</i> < 2 and generally 0.1 < <i>ŋ</i> < 1.0. For <i>ŋ</i> = 1 the error on the present pattern is completely corrected in one cycle; for <i>ŋ</i> > 1, it is overcorrected.

If the input patterns are linearly independent, the weights will converge to unique values. If not, the corrections will be oscillatory and <i>ŋ</i> should decrease over time to allow the weights to settle. One possible schedule is <i>ŋ</i> = <i>k</i><sup>-1</sup>, where <i>k</i> indexes the iterations.

The Adaline network is first tested in-depth on the AND and XOR datasets:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks/images/5.png)

As we can see, XOR is not learned (as expected), however the AND dataset is learned, and the decision boundary separating them is beginning to converge to a location such that each datapoint is as far from the boundary as possible (Mean Squared Error Loss).

Next, the network is tested on all 16 Boolean functions:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks/images/6.png)

Note how for the 14 learnable networks, the decision boundaries are converging to minimize the Mean Squared Error loss.

Finally, since Adaline networks are capable of accepting continuous inputs, we try a randomized clustered dataset with continous values:

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks/images/7.png)

As we can see by the graph, the Adaline network successfully learned how to classify each cluster correctly.


### [Continue to Section 4.1](https://github.com/jlehett/Neural-Smithing/tree/master/4.%20MLP%20Representational%20Capabilities/4.1%20Representational%20Capability)
