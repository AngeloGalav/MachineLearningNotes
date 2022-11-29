
# Preprocessing
As we mentioned before, we need to preprocess data so that the whole learning process becomes more efficient and improves in general. 

There are various kinds of preprocessing methods:
- 

## Aggregation 
It consists in combining two or more attributes (or objects) into a single attribute (or object). We could do this by, for example, reducing the data. 

This process has many purposes, such as:
- __Data reduction__: reduce the number of attributes or objects. i.e. reducing the number of records by collapsing into one. 
- __Change of scale__: cities aggregated into regions, states, countries, etc... 
- __More stable data__: aggregated data tends to have less variability (and less overfitting)

i.e. we could consider the yearly data instead of the monthly data.

## Sampling

The process of sampling is important for both preliminary investigation and final data analysis:
- Statistician perspective: obtaining the entire data set could be impossible or too expensive 
- Data processing perspective processing the entire data set could be too expensive or time consuming

We saw the randomness int the train-test split & cross validation. 
==Using a sample will work almost as well as using the entire data sets, if the sample is representative== (obv. this could never be 100% true).
In general, a sample is __representative__ if it has approximately the same property (of interest) as the original set of data.

###### Types of sampling 
1. __Simple random__: a single random choice of an object with given probability distribution 
2. __With replacement__: repetition of independent extractions of type 1 
3. __Without replacement__: repetition of extractions, extracted element is removed from the population 
4. __Stratified__: split data into several partitions according to some criteria, then draw the random samples from each partition. 
	- used when the data set is split into subsets with homogeneous characteristics 
	- the representativity is guaranteed inside each subset. 

In Scikit learn, when we do the train/test split, we are using a process of stratification since it ensures us that the sum of the characteristics(?) remains the same. 

Stratification is also important in cross-validation. 

##### Sampling with/without replacement
They are nearly equivalent if sample size is a small fraction of data set size.
With replacement, in a small population (a small subset) could be underestimated.
Sampling with replacement is much easier to implement, and it is much easier to be interpreted from a statistical point of view extractions are statistically independent.

Here's an example of the consequences of sample size:
![[sample_size_consequence .png]]

##### Sample size - Missing classification
The following graph shows the probability of sampling at least one element for each class with replacement (It is independent from the size of the data set!). 
![[missing_class_graph.png]]

Which means, the probability represents the probability of capturing an element from each class.  

This aspect becomes relevant, for example, in a supervised dataset with a high number of different values of the target.
If the number data elements is not big enough, ==it can be difficult to guarantee a stratified partitioning in train/test split or in cross–validation split==. 
i.e. if we have a dataset of N = 1000, C = 10, test–set–size = 300, cross–validation–folds = 10 the probability of folds without adequate representation of some classes becomes quite high.

When designing the training processes it is necessary to consider those aspects, and it would be better to do a 3 folds cross validation instead of 10. 

## Dimensionality

###### The curse of dimensionality
When dimensionality is very high the occupation of the space becomes very sparse. Discrimination on the basis of the distance becomes ineffective.

Experiment: random generation of 500 points and plot the relative difference between the maximum and the minimum distance between pairs of point.

#### Dimensionality reduction
- Purposes: 
	- avoid the curse of dimensionality 
	- noise reduction 
	- reduce time and memory complexity of the mining algorithms 
	- visualization 
- Techniques: 
	- principal component analysis 
	- singular values decomposition
	- supervised techniques 
	- non–linear techniques
 
### PCA (Principal Component Analysis)

Find projections that capture most of the data variation:
- Find the eigenvector of the covariance matrix 
- the eigenvectors define the new space

The new dataset will have only the attributes which capture most of the data variation. 

Essentially, we could drop a dimension by doing a projection of each point on an axis. Or even better, on an eighenvector so that we lose less variability. ![[PCA_graph.png]]
In the case of this image, we would do a projection on the e-vector. 

##### Feature subset selection 
There are also local ways to reduce dimensionality, these means that we do not compute eighenvectors or eighenvalues. 
Essentially, we just drop values that are deemed reduntant or irrelevant, and which do not contain any information useful for the analysis. 

Some of the methods of this type of selection are:
- __Bruteforce__: try all possible feature subsets as input to data mining algorithm and measure the effectiveness of the algorithm with the reduced dataset. 
	- It does require a lot time and computing power. 
- __Embedded approach__: feature selection occurs naturally as part of the data mining algorithm.
	- i.e. decision trees. We could compute a decision on which dimension to drop simply by looking at a decision tree.
- __Filter approach__: features are selected before data mining algorithm is run
- Wrapper approaches A data mining algorithm can choose the best set of attributes. Similar to brute force, but without exhaustive search. 

##### Feature creation
New features can capture more efficiently data characteristics:
- Feature extraction pixel picture with a face => eye distance, . . . 
- Mapping to a new space e.g. signal to frequencies with Fourier transform 
- New features e.g. volume and weight to density

# Data-type conversion
Many algorithms require numeric features, so categorical features must be transformed into numeric while ordinal features must also be transformed into numeric, with the addition that the order must be preserved. 

Classification requires a target with nominal values => a numerical target can be discretized.  Discovery of association rules require boolean features, a numerical feature can be discretized and transformed into a series of boolean features.

There are many possibilities of data conversion:
- Nominal to numeric: One-Hot-Encoding. 
- Ordinal to numeric: the ordered sequence is transformed into consecutive integers.
- Numeric to binary with threshold
- Discretization

# Similarity and dissimilarity
- Similarity 
	- Numerical measure of how alike two data objects are 
	- Is higher when objects are more alike 
	- Often falls in the range [0,1] 
- Dissimilarity 
	- Numerical measure of how different are two data objects 
	- Lower when objects are more alike 
	- Minimum dissimilarity is often 0 
	- Upper limit varies 
- Proximity refers to a similarity or dissimilarity

##### Euclidean distance - $L_2$
$$
dist = \sqrt{\sum^D_{d=1} (p_d - q_d)^2}
$$
Where D is the number of dimensions (attributes) and $p_d$ and $q_d$ are, respectively, the d-th attributes (components) of data objects $p$ and $q$.
Standardization/Rescaling is necessary if scales differ

##### Minkowski distance – $L_r$
$$
dist = ({\sum^D_{d=1} |p_d - q_d|^r})^{\dfrac{1}{r}}
$$
- Where D is the number of dimensions (attributes) and $p_d$ and $q_d$ are, respectively, the d-th attributes (components) of data objects $p$ and $q$.
- Standardization/Rescaling is necessary if scales differ.
- $r$ is a _parameter_ which is chosen depending on the data set and the application.

[Skipped 30 minutes]