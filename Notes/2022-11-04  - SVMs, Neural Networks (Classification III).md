# Non-linear classifiers

## Non-linear class boundaries in SVM
What if we have a situation with a non-linear distribution of data? 
![[svm_non_linear.png]]
If we have a situation like the one in the figure in the left, we can see that the there is no parameter for which an hyperplane can separate these two classes. 

In this case, the _nonlinearity of boundaries_ can be overcome with a __non–linear mapping__.
So, what we can do imagine another different space, called the __feature space__, in which a dimension is added. This space is calculated using a mathematical transformation, but in general it is done in a clever way, like in the image on the right in the example. 
![[feature_space_separated.png]]
Now, our data is linearly separable by adding a plane. 

This method is compatible with only a family of function (__kernel functions__).

##### The kernel trick
- The separating hyperplane computation requires a series of _dot product_ computations among the training data vectors.
- Defining the mapping on the basis of a particular family of functions, called _kernel functions_, or simply _kernels_, the mapping does not need to be explicitly computed, and the ==computation is done in the input space==. 
- This avoids an increase in the complexity. 

![[some_kernel_functions.png]]

Here are some examples:
![[decision_boundaries_examples.png]]

##### SVM complexity
- The time complexity is mainly influenced by the efficiency of the optimization library. 
- `libSVM` library scales from $O(D \cdot N^2)$ to $O(D \cdot N^3)$, depending on the effectiveness of _data caching_ in the library, which is data dependent.

### SVM final remarks
- _Learning_ is in general _slower_ than simpler methods, such as decision trees 
- _Tuning is necessary_ to set the parameters (not discussed here) 
- The results can be very accurate, because subtle and complex decision boundaries can be obtained 
- Are not affected by local minima 
- Do not suffer from the [[2022-11-11 -  Preprocessing and dissimilarities#The curse of dimensionality|curse of dimensionality]]: do not use any notion of distance (we will see later).
- SVMs do not directly provide probability estimates, these can be calculated using rather expensive estimation techniques. 
	- nevertheless, SVM can produce a confidence score related to the distance of an example from the separation hyperplane.


## Neural networks

Arrange many perceptron–like elements in a hierarchical structure, to overcome the limit of _linear decision boundary_. 

A neuron is a _signal processor_ with _threshold_. Meaning that we have an output only if our input reaches a threshold.
Signal transmission from one neuron to another is weighted, and _weights change over time_, also due to learning. 

The signals transmitted are modeled as real numbers. The threshold of the biological system is modeled as _a mathematic function_. If the function is continuous and differentiable, the mathematics is much easier (since the derivative can be expressed in terms of the function itself).  ^12155e

There are several functions available which capture this behavior, such as a sigmoid function (or an $arctan$).
![[functions_nn.png]]

#### Sigmoid function
![[sigmoid_function.png]]
It is called _squashing function_, since it "squashes" the values into \[0, 1\].
It Is continuous, differentiable, non–linear. And has this formula:
$$
\dfrac{1}{1+e^{-x}}
$$
The fact that it is non-linear is pretty important.  
A perceptron is linear, so the results is non satisfactory. This is because in a linear system, we have that $f (x_1 + x_2) = f (x_1) + f (x_2)$ , so if $x_2$ ==presents some noise, it is completely transferred to the output==. In general, this is not true in a non-linear system. 

==The shape of the function _can influence_ the learning speed==.

#### Feed–forward multi–layered network
- Inputs feed an _input layer_, one input node for each dimension in the training set.
- Input layer feeds (with weights) a _hidden layer_. 
- Hidden layer feeds (with weights) an _output layer_.
	- the number of nodes in the hidden layer are a parameter of the network 
	- the number of nodes in the output layer is related to number of different classes in the domain
		- one node if there are two classes 
		- one node per class in the other cases

In this way the signal flows from the input to the output, without loops. 
![[nn_example.png]]
Just like in perceptron, we add a _bias_ ($e_0 = 1$ in this case) so that if the input is 0, the output is not 0. 
$e_1$ and $e_2$ are the 2 variables. 
In each of the node of the hidden layer, we have a _transfer function_ (i.e. the sigmoid), which is represented as $g$. 

Since we only have one output, we have a binary classifier in this case (so it can recognize between 2 classes). 
If we want to distinguish 4 classes, we need to add an output neuron.   
We could decide instead to have 4 output nodes for 4 classes (difference between encoding or nodes for each possible output). 

If we add a node to an hidden layer into the network, we would have more parameters, and thus more _separation_.  The more complex the hidden layer -> the more power to distinguish the classes. 

#### Training a NN
The algorithm for training neural networks is similar to something like this:
```JS
set all weights to random values
while termination condition is not satisfied do
	for each training instance x do
		feed the network with x and compute the output nn(x)
		compute the weight corrections for nn(x) − xOut // output - desired out
		propagate back the weight corrections
```

The distance between the desired output and the actual output that we obtain is __propagated back__, considering the transfer function. 

Possible termination conditions are:
- checking the [[2022-10-28 - Evaluation of performance of a classifier (Classification II)#Evaluation of a classifier#Accuracy of a classifier|quality measures]].
	- checking the f-measure.
	- checking the accuracy.
- using a maximum number of iterations. 
- We should also evaluate the change of the weights: if it is small, then we stop the iteration. 

Each training loop is called an _epoch_. Sometimes, an epoch can require minutes or hours, so we should terminate the loop in this case. 

##### Remarks on NNs Training
- In NNs, _the weights encode the knowledge_ given by the supervised examples. The encoding is _not easily understandable:_ it looks like a structured set of real numbers. 
- Convergence is not guaranteed. 


### Computing the error
- Let $x$ and $y$ be the _input vector_ and the _desired output_ of a node, respectively.
- Let $w$ be the _input weight vector_ of a node.
- The error is:
$$
E(w) = \dfrac{1}{2}(y - Transfer(w,x))^2
$$
($Transfer()$ is _transfer function_).

Depending on the data, my error function can be:
- Convex: with a single global minima
- Non-convex: with many local minimas, and in this case reaching the gloabal minima is not so easy.
![[error_functions.png]]
To ==move towards the local minimum of the error function==, we need to _compute 
the [gradient](https://en.wikipedia.org/wiki/Gradient_descent)_, and follow it. 
To compute the gradient, we need to compute the derivates of the error function w.r.t. the weights. 
![[error_function.png]]
We can see that the derivative of the sigmoid is computed using the function itself, and [[#^12155e|as we've said, this eases the computation]]. 
By computing the derivatives in respect to the weights, we know in which direction to move.  

The weight is changed subtracting ==the partial derivative== multiplied by a __learning rate__ constant. 
- The learning rate ==influences the convergence speed== and can be adjusted as a _tradeoff between speed and precision_.
- The subtraction moves towards smaller errors. 
- The derivatives of the input weights of the nodes of a layer can be computed ==if the derivatives for the following layer are known==.

The algorithm to train neural networks could be rewritten in this way:
```JS
set all weights to random values
	while termination condition is not satisfied do
		for each training instance x do
			1 – feed the network with x and compute the output nn(x)
			2 – compute error prediction at output layer nn(x) − xOut
			3 – compute derivatives and weight corrections for output layer
			4 – compute derivatives and weight corrections for hidden layer
```
Steps 1 and 2 are _forward_ (computation), steps 3 and 4 are _backward_ (adjustment).
- computation -> adjustment, computation -> adjustment, ...

#### Learning modes
There are several variants for learning modes in machine learning. 

__Stochastic__ 
-  each forward propagation _is immediately followed_ by a weight update (as in the algorithm of previous slide).
- _introduces some noise_ in the gradient descent process, since the gradient is computed from a single data point. It can potentially slow the learning process. 
- reduces the chance of getting stuck in a local minimum.
- good for online learning (meaning that I want to adjust my network while the data is arriving). 

__Batch__ 
-  _many propagations_ occur before updating the weights, _accumulating errors_ over the samples within a batch
-  generally yields faster and stable descent towards the local minimum, since the update is performed in the direction of the average error

##### Some remarks
- A learning round over all the samples of the network is called epoch
- In general, after each epoch _the network classification capability will be improved_ 
- Several epochs will be necessary 
- After each epoch the starting weights will be different
- __Overfitting is possible__ w.r.t. the complexity of the decision problem.
	![[overfitting_nn.png]]
A situation like the one seen in this picture will not generalize well. 

##### Design choices 
Here some design choices we can make during the design process of a NN.   
- The structure of input and output layers _is determined by the domain_ (the training set), i.e. the number of columns.
- The number of nodes in the hidden layer can be changed, or I can have multiple hidden layers. 
- The ==learning rate can be changed in different epochs==
	- in the beginning a higher learning rate can push faster towards the desired direction
	- in later epochs a lower learning rate can push more precisely towards a minimum

##### Regularization
As we've said, overfitting is possible, but we can use __regularization__ to improve the generalization capabilities of the model. 
What we do is modify the performance function, which is normally chosen to be the sum of squares of the network errors on the training set.
_Improvement of performance_ is obtained by _reducing a loss function_ (i.e. the sum of squared errors). Regularisation corrects the loss function in order to smooth the fitting to the amount of data. Obviously, regularisation must be tuned. 

##### NNs application example 
- Recognition of characters with noise 
- The images are 7x5 arrays of floating points (before noise -1 for black, 1 for white) 
- 35 input nodes 
- up to 60 hidden nodes 
- 26 output nodes 
	- only one at a time should give a value near 1, all the others should be near 0 
	- a good alternative could be a 5 bit coding
![[nn_graph_example.png]]
With a noisy signal, the performance of the model is bad in every case. 
(A good rule of thumb: you should a number of nodes that is almost double the nodes of the output). 

What we learn from this is that NNs behave very well in presence of noise. 

## K-nearest neighbors classifier (or KN, KNN). 
While the other methods encode the knowledge in a model (i.e. DT or NNs), ==this method keeps all the training data== (i.e. the model is the entire training set). 
It is the basis of clustering. 

- Future predictions by ==computing the similarity between the new sample and each training instance== (by using for example any similarity function)
- picks the ==K entries in the database which are closest to the new data point==. 
- ==majority vote==
- main parameters: 
	- the number of neighbors to check
	- the metric used to compute the distances (the Mahalanobis distance has good performance)

Essentially, I go on in checking each neighbor, and by checking the nearest neighbor to each instance, I decide the class of the new individual.

## From binary classifier to multiclass classification. 
As we've seen, several classifiers generate a binary classification model. 
There are two ways to deal with multi–class classification:
- transform the training algorithm and the model, sometimes at the expenses of an increased _size_ of the problem (not important)
- _use a set of binary classifiers_ and _combine the results_, at the expenses of an increased _number_ of problems to solve. 
	- one-vs-one and one-vs-rest strategies

### One-vs-one strategy (OVO)
Consider that we've 3 classes, for example A1, A2, A3. We would need to build a classifier that distinguishes between (A1,A2), (A2,A3) and (A3,A1). 

So, consider all the possible pairs of classes and generate a _binary classifier for each pair_.  
(we'll have $C * (C - 1)/2$ pairs). 
Each binary problem considers only the examples of the two selected classes. 
At prediction time _apply a voting scheme_: 
- an unseen example is submitted to the $C * (C - 1)/2$ binary classifiers, each winning class receives a $+1$ 
- the class with the highest number of $+1$ wins

### One-vs-rest strategy (OVR)
Consider $C$ binary problems where class $c$ is a positive example, and all the others are negatives. 
- build $C$ binary classifiers, and at prediction time _apply a voting scheme_ 
	- an unseen example is submitted to the C binary classifiers, obtaining a confidence score 
	- the confidences are combined and the class with the highest global score is chosen

In other words, it's like if we have 3 classifiers, 
1. A1 vs !A1
2. A2 vs !A2
3. A3 vs !A3

### OVR vs OVO
- OVO requires _solving a higher number of problems_, even if they are of smaller size. 
- OVR tends to be _intrinsically unbalanced_. If the classes are evenly distributed in the examples, each classifier has a proportion positive to negative 1 to C - 1. 

-----
Here's a small, not really important, deviation to the normal lesson. 

## Deep learning - a small introduction
Deep learning deals with deep networks, which are neural networks with a large number of hidden layers. Normal NNs are called Shallow Networks. 
 
The idea of neural networks is very old (80s), but at the time computational power was not enough to compute deep networks, but also there were numerical problems (like stability etc..). 

With the advent of GPUs, which are very complex parallel computing units, we could finally deal with millions of parameters. 

Deep networks need more parameters, so also they need large dataset. This means that while deep learning is a very nice concept, it isn't always the right method to apply.  

We could say that deep learning is an evolution of machine learning.  