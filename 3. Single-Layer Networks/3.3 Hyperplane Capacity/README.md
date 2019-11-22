## 3.3 Hyperplane Capacity

A <i>dichotomy</i> is a divison of <i>N</i> points into two classes. A dichotomy is linearly separable if all the 0s can be separated from all the 1s by a hyperplane. It is <i>homogeneously linearly separable</i> if the points can be separated by a hyperplane passing through the origin.

A linear separation of <i>N</i> points in <i>d</i> dimensions is a homogeneous linear separation in <i>d</i> + 1 dimensions because the offset of a hyperplane that does not pass through the origin can be absorbed into an extra bias weight.

The capacity of a hyperplane is the number of dichotomies it can separate. For <i>N</i> points in general position in a Euclidean space of dimensions <i>d</i>, the number <i>C</i>(<i>N</i>, <i>d</i>) of homogeneously linearly separable dichotomies is given by the following reccurence:

<i>C</i>(<i>N</i>, <i>d</i>) = SUM<sup><i>N</i>-1</sup><sub><i>k</i>=0</sub> \[ ( <i>N</i> - 1 choose <i>k</i> ) * <i>C</i>(1, <i>d</i> - <i>k</i>) \] ,

<i>C</i>(1, <i>m</i>) = { <sup>2, <i>m</i> ≥ 1</sup> <sub>0, <i>m</i> < 1
                                                                         
Running the program found in this section will ask you for the number of points <i>N</i>, and the number of dimensions <i>d</i> as input to solve for the capacity of a hyperplane given <i>N</i> points in <i>d</i> dimensions. The graphic below shows an example of this:


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.3%20Hyperplane%20Capacity/images/1.png)


As we can see, we ran this on 4 points in 2 dimensions. This is equivalent to the Boolean functions we tested before. The result of 14 makes sense in this case. A hyperplane separating 4 points in 2 dimensions will be able to separate 14 out of the possible 16 dichotomies.

Given this information, we can solve for the probability that a randomly chosen dichotomy of <i>N</i> points in <i>d</i> dimensions is linearly separable. We could simply take our hyperplane capacity, <i>C</i>(<i>N</i>, <i>d</i>), and divide this by the total number of dichotomies that are possible. The total number of dichotomies possible is computed simply as 2<sup><i>N</i></sup>.

The probability that a randomly chosen dichotomy is linearly separable is then

<i>f</i>(<i>N</i>, <i>d</i>) = { <sup>1,   <i>N</i> ≤ <i>d</i> + 1</sup><sub>2/2<sup><i>N</i></sup> * SUM<sup><i>d</i></sup><sub><i>k</i>=0</sub> \[ ( <i>N</i> - 1 choose <i>k</i> )],   <i>N</i> > <i>d</i>+1</sub>

Running the second part of the program found in this section will ask you for the number of points <i>N</i>, and the number of dimensions <i>d</i> as input to solve for the probability that a randomly chosen dichotomy is linearly separable given <i>N</i> points in <i>d</i> dimensions. The graphic below shows an example of this:


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.3%20Hyperplane%20Capacity/images/2.png)


This probability is plotted below as a function of <i>N</i> / (<i>d</i> + 1) for a few values of <i>d</i>:


![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/3.%20Single-Layer%20Networks/3.3%20Hyperplane%20Capacity/images/3.png)


When <i>d</i> is large, <i>f</i>(<i>N</i>, <i>d</i>) is greater than 1/2 as long as <i>N</i> < 2(<i>d</i> + 1) and less than 1/2 for <i>N</i> > 2(<i>d</i> + 1). As <i>d</i> becomes large, the transition becomes steeper and almost all dichotomies of <i>N</i> < 2(<i>d</i> + 1) are linearly separable while almost all dichotomies of more points are not.

When <i>N</i> < 2<i>d</i>, generalization may be poor because there is a high probability that the training points will be linearly separable even if the function generating the labels is not; the network is underconstrained and the solution is unlikely to generalize well to new patterns. In linear regression, a common heuristic is to require <i>N</i> ≥ 3<i>d</i> or more training patterns to avoid overfitting.


### [Continue to Section 3.4](https://github.com/jlehett/Neural-Smithing/tree/master/3.%20Single-Layer%20Networks/3.4%20Learning%20Rules%20for%20Single-Layer%20Networks)
