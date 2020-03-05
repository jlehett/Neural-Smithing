## 4.2 Universal Approximation Capabilities

Multilayer perceptrons are <i>universal approximators</i>. They are capable of arbitrarily accurate approximation of essentially arbitrary continuous mappings from the [-1, +1]<sup>n</i> hypercube to the (-1, 1) interval. Neural networks, as a class, are powerful enough to implement essentially any function we require.

MLP approximation error scales with the number of training samples and the number of parameters. The error decreases like <i>O</i>(1 / SQRT(<i>N</i>)) as the number of training samples <i>N</i> increases. The error decreases like <i>O</i>(1 / <i>M</i>) as a function of <i>M</i>, the number of hidden nodes. 

These results provide another justification for the rule of thumb that the number of training samples should be larger than the number of parameters divided by the desired approximation error, <i>N</i> > <i>O</i>(<i>M p </i>/ <i>ϵ</i>). Here, <i>N</i> is the number of samples, <i>M</i> is the number of hidden nodes, <i>p</i> is the input dimension (so <i>M p</i> is approximately the number of parameters), and <i>ϵ</i> is the desired approximation error. 

This relationship is coded up in the Python file included in this directory. With this script, you can find the number of training samples that should be used to train a classifier given the input dimensions, the number of training examples, and the desired approximation error.

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.2%20Universal%20Approximation%20Capabilities/images/1.png)

The script also solves for the relationship showing the minimum number of hidden nodes that should be used to train a classifier given the input dimensions, the number of training samples, and the desired approximation error.

![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/4.%20MLP%20Representational%20Capabilities/4.2%20Universal%20Approximation%20Capabilities/images/2.png)

### [Continue to Section 4.3](https://github.com/jlehett/Neural-Smithing/tree/master/4.%20MLP%20Representational%20Capabilities/4.3%20Size%20Versus%20Depth)
