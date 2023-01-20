There are 2 types of classification: supervised and unsupervised classifications. 
As of now, we're only interested in __supervised__ classifications. 

# Supervised Classification
Consider the "soybean" example shown in the introduction ([here](https://virtuale.unibo.it/pluginfile.php/1336016/mod_resource/content/7/machineLearning-01-intro.pdf), slides 21). 
The data set $X$ contains $N$ individuals described by $D$ attribute values each. 
We have also a $Y$ vector which, for each individual $x$ contains the _class_ value $y(x)$.
The class allows a finite set of different values (e.g. the diseases), let's call it $C$.
The class values are provided by experts: _the supervisors_. 

We want to learn ==how to guess the value of the $y(x)$== for individuals which have not been examined by the experts. 
We want to learn a __classification model__.

## Classificiation Model  
A classification model is an algorithm which, given an individual for which the class is not known, ==computes the class==. 
The algorithm is _parametrized_ in order to optimize the results for the specific problem at hand. 

Developing a classification model requires:
- choose the _learning algorithm_ 
- let the algorithm learn its parametrization (through the use of already supervised data)
- assess the quality of the classification model 

The _classification model_ is used by a run–time __classification algorithm__ with the developed parametrization. This means that we'll use the model to classify new data. 

Usually, there will also be an additional parameter (called $\theta$, usually it is a threshold) that will influence the output. We'll see shortly. 

### Classification model or, shortly, _classifier_
A _classifier_ is a decision function which, given a data element $x$ whose class label $y(x)$ is unknown, makes a prediction as
$$
M(x, \theta) = y(x)_{pred}
$$
where $\theta$ is a set of values of the parameters of the decision function.
The prediction can be true or false. 

The learning process for a given classifier $M(., .)$, given the dataset $X$ and the set of _supervised class labels_ $Y$, determines $θ$ ==in order to reduce the prediction error as much as possible==. 

```
Why am I not able to reduce the prediction error to 0? The answer is that it is because I cannot predict all the possible situations in advance. In the real world the information can be slightly or no 
The second reason is that my decision function will never be able to learn every possible decision function. 
```

### Example of a decision function
![[decision_function_example.png]]
In this example, we have a dataset with two dimensions and 2 classes. 
The color of the dot represents the class. 
One could decide a decision function as a straight line, and we'll get a function like this:
![[decision_function_example_2.png]]
This model makes some errors, and even the best choice of parameters (the $\theta$s) cannot avoid any errors.  
The phrase "all models are wrong, but some are useful" captures the fact that we cannot model reality in precision.

Nevertheless, different models can have different power to _shatter the dataset into subsets_ with homogeneous classes. In the previous example, we could use a quadratic function for example. 

### Vapnik-Chervonenkis Dimension
This property relates to the _shattering power_ of a classification model.
- Given a dataset with $N$ elements there are $2^N$ possible different learning problems.
- If a model $M(., .)$ is able to shatter _all the possible learning problems_ with $N$ elements, we say that it has _Vapnik-Chervonenkis Dimension_ equal to $N$ 
- The straight line has VC dimension 3, since i can separate for sure 3 elements with a single straight line. 
	- don’t worry, frequently, in real cases, data are arranged in such a way that also a straight line is not so bad 

## Classification workflow

1. Learning the model for the given set of classes, meaning that:
	1. a _training set is available_, containing a number of individuals 
	2. for each individual _the value of the class label is available_ (also named ground truth)
	3. the _training set should be representative_ as much as possible (the training set should also be obtained by a random process)
		- Meaning that the training data should be highly comparable with the data used in the test set. 
	4. the model is fit learning from data the best parameter setting

2. Estimate the accuracy of the model (the number of good predictions made on the test set). 
	1. a _test set_ is available, for which the class labels are known. 
		- Essentially, I used my supervised data and I split it into 2 parts: a test set and a training set. 
	2. the model is run by a classification algorithm to assign the labels to the individuals on the test set. 
	3. ==the labels assigned by the model are compared with the true ones==, to estimate the accuracy. 
	
3. The model is used to label new individuals.

![[classification_workflow.png]]

### Two flavor classification
- __Crisp__: the classifier assigns to each individual _one_ label.
- __Probabilistic__: the classifier assigns a _probability for each of the possible labels_.

# Decision Trees
A run–time classifier structured as a decision tree is a _tree–shaped set of tests_. 
The decision tree has inner nodes and leaf nodes:
- Inner nodes: `if test on attribute d of element x then execute node1 else execute node2`
- Leaf nodes: `predict class of element x as c2`

## Generating this model
Given a set $X$ of elements for which the class is known (supervised data), grow a decision tree as follows:
- if all the elements belong to class $c$ or "$X$ is small" generate a leaf node with label $c$ 
- otherwise 
	- choose a test based on a _single attribute with two or more outcomes_.
	- make this test the root of a tree with one branch for each of the outcomes of the test 
	- _partition_ $X$ _into subsets_ corresponding to the outcomes and _apply recursively the procedures to the subsets_.

![[decision_tree_ex.png]]

Essentially: if we can stop, fine, otherwise we must partition the dataset. In the beginning of the tree, I have an entire dataset. According to the outcome of the inner node I split the dataset into two parts.

This recursive procedure allows us to reduce the _search space_. 

Note: in the example [[decision_function_example.png|here]] we are testing 2 attributes at the same time. If we were to test only a single attribute, then we would be using a line which is parallel to one of the 2 main axis. 

### Problems to solve
1. which attribute should we test?
2. which kind of tests? (binary, multi–way, ... , depends also on the domain of the attribute)
3. what does it mean "$X$ is small", in order to choose if a leaf node is to be generated also if the class in X is not unique?

### Pairplots
A _pairplot_ is a two-dimensional representation of a dataset, which can be seen as a collection of scatterplots.  

The diagonal shows the distribution of values for a single attribute. while each cell shows us the so-called _scatterplots_. In each scatterplot, each point is associated with 2 attributes, while the color represents the class. 
![[pairplot_example.png]]
It is considered one of the most powerful visual representation of a dataset. 

## Supervised learning goals
Our goal is to design an algorithm to find interesting patterns (in classification, we mean to find a label), ==in order to forecast the values of an attribute given the values of other attributes==.
Also:
- we need to distinguish real patterns from illusions (not all patterns are interesting).
In our case, we must find patterns to __guess the class__ given the other values.

How much can we evaluate if a pattern is interesting?
There are several methods, one of them is based on information theory, which is primarily used in telecommunication, and is based in the concept of _entropy_.
- Also, evaluating how much a pattern is interesting is like trying to find the best threshold to 'split' the dataset (i.e. splitting the dataset according to the value of an attribute)

### Entropy
To understand this concept, let's start with an example:
- Given a variable with 4 possible values and a given probability distribution $P(A) = 0.25$, 
$P(B) = 0.25$, $P(C) = 0.25$, $P(D) = 0.25$ 
- an observation of the data stream could return `BAACBADCDADDDA...`

I could decide to use a 2-bit encoding: 
$$
A = 00, B = 01, C = 10, D = 11
$$
Converting the transmission will be very easy (it will be `0100001001001110110011111100`).

##### But...
When the probability distribution are uneven, the problem becomes quite complex.
Let's consider:
$$
P(A) = 0.5, P(B) = 0.25, P(C) = 0.125, P(D) = 0.125
$$

We could use the same encoding as the first example, but we can do better.
Is there a coding requiring only 1.75 bit per symbol, on the average? YES!
$A = 0, B = 10, C = 110, D = 111$

##### We can improve it tho!
What if there are only three symbols with equal probability? $P(A) = 1/3, P(B) = 1/3, P(C) = 1/3$
Again, the two–bit coding shown above is still possible. 
But is there a coding requiring less than 1.6 bit per symbol, on the average? YES!
$A = 0, B = 10, C = 11$ or any permutation of the assignment

#### General case
We can generalize what has been just said using the following considerations.
Given a source $X$ with $V$ possible values, with probability distribution:
$$
P(v_1) = p_1, P(v_2) = p_2, \dots, P(v_v) = p_v  
$$
the best encoding allows the transmission with an average number of bits given by:
![[entropy_formula.png]]
$H(X)$ is the __entropy__ of the information source $X$. 

Spoiler: we will consider as information source the _information of classes_ of each individual, so the different values will be the different labels of the classes. 

What happens if we have only 1 symbol?
Then, some probabilities will be 0 and one will be 1, and the $H(x)$ will be 0. _That means that if we have 0 variability, we have 0 entropy_. 

### Meaning of entropy
In general:
- ==__High entropy__ means high uncertainty==, and that _the probabilities are mostly similar_. So the histogram would be flat. 
- __Low entropy__ means low uncertainty, so _some symbols have much higher probability_ (wrt others). Our histogram will have peaks. 
- A higher number of allowed symbols increases the entropy.
![[entropy_histogram.png]]
In a binary source with symbol probabilities p and (1-p) when p is 0 or 1, the entropy goes to 0. This can be seen clearly in the following image:
![[entropy_formula_binary.png]]
###### Entropy after a threshold–based split
Our objective is to use a decision function in order to split the dataset in two parts according to a threshold on a numeric attribute. 
Thus, the entropy changes, and becomes the weighted sum of the entropies of the two parts. 
The weights are the relative sizes of the two parts.

Let $d \in D$ be a real-valued attribute, let $t$ be a value of the domain of $d$, let $c$ be the class attribute. 

We define the entropy of $c$ with respect to $d$ with threshold $t$ as:
$$
H(c|d:t) = H(c|d<t)*P(d<t) + H(c|d \ge t) * P(d \ge t)
$$
which means that is the sum between:
- The entropy of $c$ for individuals where $d$ is less than the threshold $t$, multiplied by the probability of those individuals.  
- The entropy of $c$ for individuals where $d$ is more than the threshold $t$, multiplied by the probability of those individuals. 

In general, the new entropy will be smaller than the previous one. 

### Information gain for binary split
It is the _reduction of the entropy of a target class_ obtained with a _split of the dataset_ based on a threshold for a given attribute.
We define $IG(c|d:t) = H(c) - H(c|d:t)$, it is the information gain provided when we know if, for an individual, $d$ exceeds the threshold $t$ in order to forecast the class value. 

We define $IG(c|d) = max_t \ IG(c|d:t)$.
- "The information gain of $c$ making the best split on attribute $d$, is the maximum (in $t$) of the Information Gain of $c$ making the best split on attribute $d$ with threshold $t$." 

Let's consider the iris dataset for example:
![[iris_example_ig.png]]
In this case, choosing threshold $t = 1.8$ results in a pretty good split. Since we have mostly the same individuals in both cases (with a small margin of error: some of the low are on high and viceversa).  

If we plot the information gain changing the threshold, we'll see that there is a maximum at around 1.8:
![[Screenshot_20230120_153127.png]]

### What can we do with information gain?
Predict the probability of long life given some historical data on person characteristics and life style:
Ex. 
![[IG_example.png]]

The attribute that gives us the most information is the Gender. 
The attribute "LastDigitSSN" is random noise, and is negligible.

### Decision Tree generation pt. 2 
Which attribute should we test?
- test the attribute which __guarantees the maximum IG__ for the class attribute in the current data set $X$ 
- partition $X$ according to the test outcomes
- recursion on the partitioned data

The supervised dataset will be split in two parts, randomly:
- One will be used as the __training set__, used to learn the model.
- The other will be used as the __test set__, used to evaluate the learned model on fresh data.

The proportion of the split is decided by the experimenter, but the parts _must_ have similar characteristics (i.e. same proportion of attributes). 
Some of the proportions are 80-20, 67-33 and 50-50.

#### One stump decision
It's the easiest decision tree, and has only one split. 
![[decision_tree_result_example.png]]

### To sum up...
To generate a decision tree:
- choose the attribute giving the highest IG 
- partition the dataset according to the chosen attribute 
- choose as class label of each partition the majority