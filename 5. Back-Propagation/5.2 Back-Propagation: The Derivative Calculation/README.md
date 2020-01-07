## 5.2 Back-Propagation: The Derivative Calculation

Having obtained the outputs and calculated the error, the next step is to calculate the derivative of the error with respect
to the weighhts. First we note that <i>E<sub>SSE</sub></i> = <b>SUM</b> <sub><i>p</i></sub> [ <i>E<sub>p</sub></i> ] is just
the sum of the individual pattern errors so the total derivative is just the sum of the per-pattern derivatives

<sup><i>∂ E</i></sup> / <sub><i>∂ w</i><sub><i>ij</i></sub></sub> = <b>SUM</b> <sub><i>p</i></sub> [ <sup><i>∂ E<sub>p</sub>
</i></sup> / <sub><i>∂</i> <i>w<sub>ij</sub></i></sub> ]

The thing that makes back-propagation efficient is how the operation is decomposed and the ordering of the steps. The
derivative can be written

<i><sup>∂ E<sub>p</sub></sup></i> / <i><sub>∂ w<sub>ij</sub></sub></i> = <b>SUM</b> <sub><i>k</i></sub> [ <i><sup>∂ E<sub>p
</sub></sup></i> / <i><sub>∂ a<sub>k</sub></sub></i> * <i><sup>∂ a<sub>k</sub></sup></i> / <i><sub>∂ w<sub>ij</sub></i> ]

where the index <i>k</i> runs over all output nodes and <i>a<sub>j</sub></i> is the weighted-sum input for node <i>j</i>
obtained in equations 5.3. It is convenient to first calculate a value <i>𝛿<sub>i</sub></i> for each node <i>i</i>

<i>𝛿<sub>i</sub></i> = <i><sup>∂ E<sub>p</sub></sup></i> / <i><sub>∂ a<sub>i</sub></sub></i>

= <i><sup>∂ E<sub>p</sub></sup></i> / <i><sub>∂ y<sub>i</sub></sub></i> * <i><sup>∂ y<sub>i</sub></sup></i> / <i><sub>
∂ a<sub>i</sub></sub></i>

which measures the contribution of <i>a<sub>i</sub></i> to the error on the current pattern.

For output nodes, <i>∂ E<sub>p</sub></i> / <i>∂ a<sub>k</sub></i> is obtained directly

<i>𝛿<sub>k</sub></i> = - (<i>d<sub>pk</sub></i> - <i>y<sub>pk</sub></i>) <i>f '<sub>k</sub></i>

where <i>f '<sub>k</sub></i> ≡ <i>f '</i> (<i>a<sub>k</sub></i>)

For hidden nodes, <i>𝛿<sub>i</sub></i> is obtained indirectly. Hidden nodes can influence the error only through their
effect on the nodes <i>k</i> to which they send output connections so

<i>𝛿<sub>i</sub></i> = <i><sup>∂ E<sub>p</sub></sup></i> / <i><sub>∂ a<sub>i</sub></sub></i> = <b>SUM</b> <sub><i>k</i></sub>
 [ <i><sup>∂ E<sub>p</sub></sup></i> / <i><sub>∂ a<sub>k</sub></sub></i> * <i><sup>∂ a<sub>k</sub></sup></i> / <i><sub>∂ a<sub>
i</sub></sub></i> ]

But the first factor is just the <i>𝛿<sub>k</sub></i> of node <i>k</i> so

<i>𝛿<sub>i</sub></i> = <b>SUM</b> <i><sub>k</sub></i> [ <i>𝛿<sub>k</sub></i> * <i><sup>∂ a<sub>k</sub></sup></i> / <i>
<sub>∂ a<sub>i</sub></sub></i> ]

The second factor is obtained by noting that if node <i>i</i> connects directly to node <i>k</i> then <i>∂ a<sub>k</sub></i>
 / <i>∂ a<sub>i</sub></i> = <i>f '<sub>i</sub></i> * <i>w<sub>ki</sub></i>, otherwise it is zero. So we end up with
 
 <i>𝛿<sub>i</sub></i> = <i>f '<sub>i</sub></i> * <b>SUM</b> <i><sub>k</sub></i> [ <i>w<sub>ki</sub> 𝛿<sub>k</sub></i> ]
 
 Because <i>𝛿<sub>k</sub></i> must be calculated before <i>𝛿<sub>i</sub></i>, <i>i</i> < <i>k</i>, the process starts at
 the output nodes and works backward toward the inputs, hence the name "back-propagation."
 
 To summarize so far,
 
 <i>𝛿<sub>i</sub></i> =
 
 = - (<i>d<sub>pi</sub></i> - <i>y<sub>pi</sub></i>) * <i>f '<sub>i</sub></i> (for output nodes)
 
 = <i>f '<sub>i</sub></i> * <b>SUM</b> <sub><i>k</i></sub> [ <i>w<sub>ki</sub> 𝛿<sub>k</sub></i> ] (for hidden nodes)
 
 For output nodes, <i>𝛿<sub>i</sub></i> depends only on the error <i>d<sub>i</sub></i> - <i>y<sub>i</sub></i> and the local
 slope <i>f '<sub>i</sub></i> of the node activation function. For hidden nodes, <i>𝛿<sub>i</sub></i> is a weighted sum of
 the <i>𝛿</i>s of all the nodes it connects to, times its own slope <i>f '<sub>i</sub></i>.
 
 Having obtained the node deltas, it is an easy step to find the partial derivatives <i>∂ E<sub>p</sub></i> / <i>∂ w</i>
 with respect to the weights. The derivative of pattern error <i>E<sub>p</sub></i> with respect to weight <i>w<sub>ij</sub>
 </i> is then
 
 <i><sup>∂ E<sub>p</sub></sup></i> / <i><sub>∂ w<sub>ij</sub></sub></i> = <i>𝛿<sub>i</sub> y<sub>j</sub></i>,
 
 where <i>y<sub>j</sub></i> is the output activation of node <i>j</i>.
 
 The code in this section shows how to perform back-propagation in Python. It builds upon the neural network class that
 was developed in section 5.1. An example of the output of the code is shown below:
 
 ![Graphs](https://github.com/jlehett/Neural-Smithing/blob/master/5.%20Back-Propagation/5.2%20Back-Propagation:%20The%20Derivative%20Calculation/images/1.png)
 
 ### [Continue to Section 5.3](https://github.com/jlehett/Neural-Smithing/tree/master/5.%20Back-Propagation/5.3%20Back-Propagation:%20The%20Weight%20Update%20Algorithm)
