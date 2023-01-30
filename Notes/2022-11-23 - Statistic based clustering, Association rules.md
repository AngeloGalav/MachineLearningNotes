# Model based (or statistic based) clustering
The clustering models that we've seen so far are do not make assumptions on the distribution of data, so they work either consider distances or look around etc... 

Model based clustering focuses on _estimating the parameters_ of a statistical model to _maximize_ the ability of the model to _explain the data_. 

The main technique is to use the _mixture models_ (i.e. Gaussian models). So, we view the data as a set of observation from a mixture of different probability distributions. 
- i.e., we could have a series of points in a space, that could be approximated by two different Gaussians for axis x and y respectively 
- Or, we could have a series of points in a single line expressed as two Gaussian in sequence, we'll see that more clearly in [[|this example]]. 

Usually, the base model is a _multivariate normal distribution_.
The estimation is usually done using the _maximum likelihood_: given a set of data $X$, the _probability of the data_, in function of the parameters of the model, is called a _likelihood function_.
- Means: find the parameters which maximize the likelihood of the model w.r.t. the data(??). 
Attributes are assumed to be _random [[2022-11-09 - Ensemble methods, Regression#Creating independence|independent variables]]_.

\[These methods do not require the number of clusters as an input :D\]

## Gaussian Mixture a.k.a. Expectation Maximization (EM) 
If the data can be approximated by a single distribution, the derivation of the parameters is straightforward.
In the general case, with many mixed distributions, the EM algorithm is used.

Here's the actual algorithm:
1. Select an initial set of model parameters 
2. repeat:
	1. __Expectation Step__ – For each object, calculate the _probability_ that _each object belongs to each distribution_. 
		- i.e., if we have a graph with 3 Gaussians, we compute the probability of belonging in each model. 
	2. __Maximization Step__ – Given the probabilities from the expectation step, _find the new estimates of the parameters_ that _maximize the expected likelihood_. 
3. until – the parameters do not change (or the change is below a specified threshold)

\[Notice: this algorithm is very much similar to [[2022-11-16 - Last remarks on Transformations, Introduction to Clustering (K-Means)#K-means|k-means]], but in practice are completely different.\]

Here's an example. We have one dimension of data with a mixture of two models. 
![[gaussians.png]]
![[extracted_model.png]]

If we have two clusters of values, like in the example, we need to estimate _5 parameters_:
- _mean_ and _standard deviation_ for cluster A
- mean and standard deviation for cluster B 
- _sampling probability_ $p$ for cluster A (meaning, the probability of belonging to cluster A).

![[formulas_distribution.png]]

In this case, the algorithm does something like this:
- Repeat until convergence 
	- __Expectation__: Compute $p_A$ and $p_B$ using the current distribution parameters.
		- Compute the numerators for $Pr(A | x)$ and $Pr(B | x)$ and normalize dividing by their sum 
	- __Maximization__ of the distribution likelihood given the data
		- Compute the _new distributions parameters_ (i.e. $\micro$, $\sigma$...), _weighting the probabilities_ according to the current distribution parameters.  
- After convergence, _label each object_ with A or B according to the _maximum probability_, given the last distribution parameter.

Let's compare this method with K-means:
![[gaussian_vs_kmeans.png]]
On the left, we have clustering with EM, while on the right we have clustering with K-means. 

Since these data have a Gaussian–like distribution, and the EM algorithm is founded on the hypothesis of modelling data with Gaussians, it will of course be better. 
Thus, since K-Means is non–parametric, in this case its performance is worse. 

# Final remarks
![[all_clusters.png]]

### Clustering types
- _Partitioning_: iteratively find partitions in the dataset, optimizing some quality criterion.
- _Hierarchic_: recursively compute a structured hierarchy of subsets. 
- _Density based_: compute densities and aggregates clusters in high density areas. 
- _Model based_: assume a model for the distribution of the data and find the model parameters which guarantee the best fitting to the data.

### Clustering scalability 
Effectiveness decreases with 
- dimensionality D 
- noise level 
Computational cost increases with
- dataset size N, at least linearly
- dimensionality D

----
# Association Rules
It is an unsupervised kind of activity, and it used for making prediction on a transactional database. 

## Market basket example
Given a set of commercial transactions, find _rules_ that _will predict_ the _occurrence of an item_ based on the _occurrences of other items_ in the transaction.

Our table representing commercial transaction looks something like this:
![[table.png]]
It looks like a relational database table. 
- TID is the transaction id. Not useful for learning, but only for preprocessing. 

Some examples of association rules are:
- {Diaper} → {Beer}
	- Meaning, if there is a diaper, there will probably also beer.  
- {Bread, Milk} → {Coke, Eggs}
- {Beer, Bread} → {Milk}

> [!WARNING]
> These are __not__ logical rules, meaning that even if they are not true, they still hold some kind of weight or importance. Meaning, association rules are _quantified_, and not just true or false . 
> For this reason, we consider the _strength_ of each rule. 

### Some definitions
- __Itemset__: a collection of one or more items 
	- Example: {Bread, Diaper, Milk}
- __k-itemset__: an itemset that contains $k$ items 
- __Support count__ ($σ$): _frequency of occurrence_ of an itemset
	- E.g. $σ$({Bread, Diaper, Milk}) = 2 
- __Support__: fraction of transactions that contain an itemset
	- E.g. $σ$({Bread, Diaper, Milk}) = 2/5 
- __Frequent Itemset__: an itemset whose support is greater than or equal to a _minsup_ threshold 
	- (simply, an itemset bigger than a threshold)

- __Association Rule__: an expression of the form A => C, where A and C are itemsets 
	- A = Antecedent and C = Consequent 
	- Example: {Diaper, Milk} -> {Beer} 
- __Rule Evaluation Metrics__: 
	- _Support_ (sup): fraction of the $N$ transactions that _contain both_ A and C.
	- _Confidence_ (conf): measures _how often_ all the items _in C_ _appear in transactions_ that _contain A_.
![[support_confidence.png]]

Confidence is always equal to the support or bigger \[he said bigger but I think he meant lower\].  

#### Some considerations
- Rules with _low support_ can be generated by _random associations_ 
- Rules with _low confidence_ are _not really reliable_ 
- Nevertheless a rule with relatively low support, but high confidence can represent an uncommon but _interesting_ phenomenon.
- i.e. caviar and champagne. It has small support, but big confidence, since they are not bought often, but when they are, they are always bought together. 

## Association Rule Mining Task 
Given a set of transactions $N$, the _goal_ of association rule mining is to find all rules having:
- support $\ge$ _minsup_ threshold
- confidence $\ge$ _minconf_ threshold
These are the rules we deem interesting. 

So, the next step will be to find the thresholds. 
A possible approach would be the brute-force approach. 
- __Brute-force approach__: 
	- List all possible association rules 
	- Compute the support and confidence for each rule 
	- Prune rules that fail the minsup and minconf thresholds

![[mining_association_rules.png]]
All the rules represents all the possible rules that we can obtain with the itemset 
{Beer, Diaper, Milk}, given our table. 
Notice that ==rules originating from the same itemset have identical support== but can have ==different confidence==
- we may decouple the support and confidence requirements.

This idea allows us to create a sort of algorithm for association rules:
- Two-step approach: 
	1. _Frequent Itemset Generation_: 
		- Generate all itemsets whose support is greater than minsup. 
	2. _Rule Generation_:
		- Generate high confidence rules from each frequent itemset, where each rule is a binary partitioning of a frequent itemset.
	
But, the _frequent itemset generation_ is still _computationally expensive_, we need to reduce the complexity of this procedure. 

## Frequent Itemset Generation
Given D items, there are $M = 2^D$ possible candidate itemsets.
The idea is to _reduce_ the _number of candidates_ $M$, by using pruning techniques. 
Or, we can reduce the number of comparisons $NM$.

#### Reducing Number of Candidates
__Apriori principle__: if an _itemset is frequent_, then _all of its subsets_ must also be _frequent_.
![[a_priori_principle.png]]
This means that the _support of an itemset_ never exceeds _the support of its subsets_. 
