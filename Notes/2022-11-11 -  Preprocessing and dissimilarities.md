
# Preprocessing
[[2022-10-19 -  The Data|As we mentioned before]], we need to preprocess data so that the whole learning process becomes more efficient and improves in general. 

There are various kinds of preprocessing methods:
- [[#Aggregation|Aggregation]] 
- [[#Sampling|Sampling]] 
- [[#Dimensionality|Dimensionality]] 
- Reduction 
- Feature subset selection 
- Feature creation
- Discretization and Binarization Attribute Transformation

## Aggregation 
It consists in combining two or more attributes (or objects) into a _single attribute_ (or object). We could do this by, for example, reducing the data, or in a more general point of view, to change scale. 

This process has many purposes, such as:
- __Data reduction__: reduce the number of attributes or objects. i.e. reducing the number of records by collapsing into one. 
- __Change of scale__: cities aggregated into regions, states, countries, etc... 
- __More stable data__: aggregated data tends to have less variability (and _less overfitting_)

i.e. we could consider the yearly data instead of the monthly data.

## Sampling
The process of sampling is important for both preliminary investigation and final data analysis:
- Statistician perspective: obtaining the entire data set could be impossible or too expensive. 
- Data processing perspective: processing the entire data set could be too expensive or time consuming.

We saw sampling in the [[2022-10-28 - Evaluation of performance of a classifier (Classification II)#Cross-validation|train-test split & cross validation]]. 
==Using a sample will work almost as well as using the entire data sets, if the sample is representative== (obv. this could never be 100% true).
In general, a sample is __representative__ if it has approximately the same property (of interest) as the original set of data.
- Same property like same distribution of values in each feature. 

###### Types of sampling 
1. __Simple random__: a single random choice of an object with given probability distribution. 
2. __With replacement__: repetition of independent extractions of type 1
	- We already saw that [[2022-11-09 - Ensemble methods, Regression#Creating independence|when we've mentioned Bagging]].  
3. __Without replacement__: repetition of extractions, _extracted element is removed_ from the population.
4. __Stratified__: split data into several partitions _according to some criteria_, then draw the _random samples from each partition_. 
	- used when the data set is split into subsets with homogeneous characteristics 
	- the _representativity is guaranteed inside each subset._ 

Stratification is also important in [[2022-10-28 - Evaluation of performance of a classifier (Classification II)#Cross-validation|cross-validation]], since the split should be done guaranteeing stratification as much as possible.  

Usually, there's a tradeoff between sampling and precision. 

##### Sampling with/without replacement
Sampling with and without replacement are _nearly equivalent_ if sample size is a small fraction of data set size.
With replacement, in a small population (a small subset) could be underestimated.
Sampling with replacement is much easier to implement, and it is much easier to be interpreted from a statistical point of view extractions are statistically independent.

Here's an example of the consequences of sample size:
![[sample_size_consequence .png]]

##### Sample size - Missing class
The following graph shows the probability of sampling at least one element for each class with replacement (It is independent from the size of the data set!). 
![[missing_class_graph.png]]
For example, let's say we have 10 classes: A B C D E F G H I J.
The graph above represents the probability of capturing at least one element from each class.  

We can see that if we draw a sample of 60 elements, the probability gets very close to 1. 
The strange thing is that this probability _does not depend on the size of the dataset_.  

This aspect becomes relevant, for example, in a supervised dataset with a high number of different values of the target.
If the number data elements is not big enough, ==it can be difficult to guarantee a stratified partitioning in train/test split or in cross–validation split==. 
i.e. if we have a dataset of N = 1000, C = 10, test–set–size = 300, cross–validation–folds = 10 (meaning that in each fold we have 30 elements) the probability of folds without adequate representation of some classes becomes quite high.

When designing the training processes it is necessary to consider those aspects, and it would be better to do a 3 folds cross validation instead of 10. 

## Dimensionality
###### The curse of dimensionality
- When dimensionality is very high the occupation of the space becomes very sparse. 
- _Discrimination on the basis of the distance becomes ineffective_.

To better understand, let's consider a random generation of 500 points and plot the relative difference between the maximum and the minimum distance between pairs of point.
![[curse_dimensionality_graph.png]]
In the x axis we have the number of the dimensions, while in the y axis we measure the ratio between the maximum distances of points in a randomly generated dataset (meaning $\dfrac{max_d - min_d}{min_d}$). 
We obtain a value that it's near 1 when minimum is very small and near 0 when the minimum is very high. 
- A value near 1 means that it's to distinguish _the points that are near_ from _the points that are far away_ (or, in other words, the range of distances becomes very very small). 

The value $\dfrac{max_d - min_d}{min_d}$ _decreases_ when _the number of dimensions increases_, so it means that it becomes more difficult to distinguish the points that are near from the points that are far away. 
In this situation, the [[2022-11-04  - SVMs, Neural Networks (Classification III)#K-nearest neighbors classifier (or KN).|k-nearest neighbors]] method becomes non-effective, and thus it is impossible to reason with distances. 

## Dimensionality reduction
To solve this problem, we can reduce the dimensionality of values through __dimensionality reduction__.  
- Purposes: 
	- avoid the curse of dimensionality 
	- _noise reduction_ 
	- _reduce time and memory complexity_ of the mining algorithms 
	- visualization 
- Techniques: 
	- principal component analysis (PCA)
	- singular values decomposition (SVD)
	- others...
 
### PCA (Principal Component Analysis)
This is perhaps the most important method in dimensionality reduction. This is an _unsupervised technique_, meaning that it is independent from the classes.  

Let's consider the following figure:
![[PCA_graph.png]]
Suppose that we have a 2-dimensional dataset where the data is distributed as in the figure.
If we drop the x1 dimension, data will be _projected_ on the x2 axis, and viceversa.
By dropping a dimension we lose some information, but if I project the data on line $e$, I lose less information (in particular, I lose less variability in comparison). 

__Principal Component Analysis__ is based in this exact notion, and its objective is to determine the _projection which allows to capture the most variability_. 
"Find projections that capture most of the data variation". In mathematical terms, we'll have to:
- Find the eigenvector of the covariance matrix 
- the eigenvectors define the new space

The new dataset will have only the attributes which capture most of the data variation. 

### Feature subset selection 
There are also _local ways_ to reduce dimensionality, which do not need to compute eigenvectors or eigenvalues:
- Essentially, we just drop values that are deemed _redundant_ or _irrelevant_ (i.e. identification numbers), and which do not contain any information useful for the analysis. 

Some of the methods of this type of selection are:
- __Bruteforce__: try all possible feature subsets as input to data mining algorithm and measure the effectiveness of the algorithm with the reduced dataset.
	- I try all the possible configurations of attributes, and seek for the best performance. 
	- It does require a lot time and computing power. 
- __Embedded approach__: feature selection occurs naturally as part of the data mining algorithm.
	- i.e. decision trees. We could compute a decision on which dimension to drop simply by looking at a decision tree.
	- We only use the attributes that have been used in the actual decision tree. 
- __Filter approach__: features are selected before data mining algorithm is run
- Wrapper approaches: a data mining algorithm can choose the best set of attributes.
	- \[the prof skipped them\]

Here's an architecture for feature (attribute) subset selection:
![[subset_selection_workflow.png]]

### Feature creation
Why would I want to create more features? The answer becomes clear when we consider, for example, the face recognition in smartphone images. Since pixel-by-pixel recognition is very much slow, some of the facial features are extracted from the face and used as attributes. 

New features can capture more efficiently data characteristics:
- Feature extraction 
	- pixel picture with a face => eye distance, . . . 
- Mapping to a new space 
	- e.g. signal to frequencies with Fourier transform 
- New features 
	- e.g. volume and weight to density

# Data-type conversion
Many algorithms require numeric features, so _categorical features_ must be transformed into _numeric_, ordinal features must also be transformed into numeric, with the addition that the order must be preserved. 

Classification requires a target with nominal values => a numerical target can be discretized.  Discovery of association rules require boolean features, a numerical feature can be discretized and transformed into a series of boolean features.

![[discretization_example.png]]

#### One-Hot-Encoding (Nominal2Numeric)
A feature with $V$ unique values is substituted by $V$ binary features each one corresponding to one of the unique values. 
If object $x$ has value $v$ in feature $d$ then the binary feature corresponding to $v$ has `True` for $x$, all the other binary features have value `False` 
- True and False are represented as 1 and 0, therefore can be processed by also by procedures working only on numeric data.

#### Ordinal2numeric
- The _ordered sequence_ is transformed into _consecutive integers_.
By default the lexicographic order is assumed. 
The user can specify the proper order of the sequence.

#### Numeric2Binary with threshold
- Not greater than the threshold becomes zero 
- _Greater_ than the threshold becomes _one_

#### Discretization
This is less strong than binarization. 
- Some algorithms work better with categorical data. 
- A small number of distinct values:
	- let the algorithms to be less influenced by noise and random effects.
	- let patterns emerge more clearly. 

Discretization can happen in two ways:
- Continuous => discrete: we use many thresholds. 
- Discrete with many values => Discrete with less values: we use some domain knowledge. 

![[discretization_example_graphs.png]]
In this graphs, we can see how the data in the $y$ axis is uniform, while the data on the $x$ axis is not (we have groupings). We discretize it in the following ways:
- In the first case, we have many different thresholds which delimit the area of a label. But, this may ignore some data or patterns. 
- Equal frequency means that the thresholds such that in each interval there will be the same amount of points. 

# Similarity and dissimilarity (proximity functions)

- __Similarity__ 
	- _Numerical measure_ of _how alike_ two data objects are 
	- Is higher when objects are more alike 
	- Often falls in the range \[0,1\] 
- __Dissimilarity__ 
	- Numerical measure of _how different_ are two data objects 
	- Lower when objects are more alike 
	- Minimum dissimilarity is often 0 
	- _Upper limit varies_ 
- Proximity refers to a similarity or dissimilarity

Here's a table which shows how dissimilarity and similarity are computed given the attribute type. $p$ and $q$ are the values of an attribute for two data objects.
![[similarity_diss_attribute_type.png]]

### Continuous values 
If our dataset is represented by continuous values, then we are in the domain usually covered by the Euclidean distance. 

##### Euclidean distance - $L_2$
$$
dist = \sqrt{\sum^D_{d=1} (p_d - q_d)^2}
$$
Where $D$ is the number of dimensions (attributes) and $p_d$ and $q_d$ are, respectively, the $d$-th attributes (components) of data objects $p$ and $q$.
Standardization/Rescaling is necessary if scales differ.
- If an attribute is in the order of magnitude that is way bigger than the others, then _it will influence the distance more_ (and in turn, if the values of this attributes are affected by noise _noise will influence the final value even more_).   
- For this reason, it is extremely common to rescale the values, using for example a MinMax scaler. 
- Some machine learning algorithms _are sensitive to feature scaling_ while others are virtually invariant to it.

(N.B.: in DTs, we dont need to scale since we have entropy. Liner perceptrons are influenced by different scales, and so are SVMs, and k-nearest neighbors). 

##### Minkowski distance – $L_r$
$$
dist = ({\sum^D_{d=1} |p_d - q_d|^r})^{\dfrac{1}{r}}
$$
- Where D is the number of dimensions (attributes) and $p_d$ and $q_d$ are, respectively, the $d$-th attributes (components) of data objects $p$ and $q$.
- Standardization/Rescaling is necessary if scales differ.
- $r$ is a _parameter_ which is chosen depending on the data set and the application.
	- if $r=1$, we have the Manhattan distance. 
		- It is used when data is binary (i.e. when data is computed in bits)
		- it works better than euclidean distance with high-dimensional data. 
	- if $r =\infty$, we have the Chebyshev distance, $\max_{d} |p_d - q_d|$.

##### Mahalanobis Distance
It is strictly related to PCA. 
The __Mahalanobis__ distance between two points $p$ and $q$ _decreases_ if, keeping the same Euclidean distance, the _segment connecting the points_ is _stretched_ along _a direction_ of _greater variation_ of data. 
The distribution is described by the _covariance matrix_ of the data set:
![[mahalanobis_distance.png]]

This notion is explained more effectively using a picture:

Consider the 3 points A,B,C, which have equal euclidean distance as shown in the picture. 
![[mahasuca_distance_graph.png]]
![[distance_.png]]
The Mahalanobis distance between A and B is 2.23.. because in that direction there is less variability. 

### Covariance matrix
- Variation of pairs of random variables
- The summation is over all the observations
- The main diagonal contains the variances
- The values are positive if the two variables grow together
- If the matrix is diagonal the variables are non–correlated
- If the variables are standardized the diagonal contains “one”
	- Standardised, meaning they are mapped onto a standard
- If the variables are standardised and non correlated, the matrix is the identity and the Mahalanobis distance is the same as the euclidean

##### Common properties of a distance
![[properties_of_a_distance.png]]

##### Common properties of similarities 
1. $Sim(p, q) = 1$ only if $p = q$
2. $Sim(p, q) = Sim(q, p)$

##### Similarity between binary vectors
![[similarity_binary_vecs.png]]

##### Cosine similarity
![[cosine_similarity.png]]
It is used in word processing.
The cosine becomes 1 when they're equal.  

#### Which proximity measure to choose?
It depends on data. 
- Dense, continuous data: a metric measure, such as the euclidean distance 
- Sparse, asymmetric data: cosine

## Correlation
![[correlation.png]]
![[correlation_2.png]]
In the picture below on this slide, we see what the correlation values are in the various situations (aka different points distribution).
As we can see, a line has _maximum correlation_. So, correlation is useful, but only when the relationship is linear.  

### Correlation between nominal attributes 
Consider the case in which we have the city of birth and the city of residence of a person. We can find a correlation between these two values, called the __Symmetric Uncertainty__.  
![[nominal_correlation.png]]
This correlation is computed using the notion of [[2022-10-21 - Classification I#Entropy|entropy]] $H()$. 
- $H(,)$ is the joint entropy, computed from the joint probabilities. 
- The entropy is 0 when $H(p) + H(q) = H(p, q)$.

Let's see an example to understand this better:
- Let's consider a SU for two independent uniformly distributed discrete attributes, say $p$ and $q$. In a variable fraction of records, the value of $p$ is copied to $q$. 

The SU will output this graph:
![[symmetric_uncert_graph.png]]
- The graph goes from complete independence (left) to complete biunivocal correspondence (right).
- when _there is independence_, the joint entropy is the sum of the individual entropies, and _SU is zero_.
- when there is _complete correspondence_, the individual entropies and the joint entropy are equal and _SU is one_. 

## Gradient descent algorithms and transformations
Machine learning algorithms that use _gradient descent_ as an _optimization technique_ require ==data to be scaled==.
- e.g. linear regression, logistic regression, neural network, etc. 
- The presence of feature value $X$ in the formula _will affect the step size_ of the gradient descent. So the bigger is the scale, the bigger is the step in that direction.
- The difference in ranges of features will cause different step sizes for each feature. 
- Similar ranges of the various features ensure that the gradient descent moves smoothly towards the minima and that the steps for gradient descent are updated at the same rate for all the features

Some transformation strategies:
![[types_of_transformations.png]]
changing the distribution values means also changing the histograms for example. 
- General mapping can change the distribution, while Standardization and MinMax don't. 