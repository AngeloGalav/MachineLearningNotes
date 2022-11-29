# Ensemble methods
Ensemble methods (aka Classifier combination) is based on training a set of __base classifiers.__
The final prediction is obtained taking the votes of the base classifiers.
In general, ensemble methods tend to perform better than a single classifier.

##### Why should this be better?
Let us consider 25 binary classifiers, each of which has error rate $e = 0.35$.

The ensemble classifier output is the majority of the predictions (13 or more). 
- If the base classifiers are equal, the ensemble error rate is still $e$.
- If the base classifiers have all error rate $e$, but they are independent, i.e. their errors are uncorrelated, then the ensemble will be wrong only when the majority of the base classifier is wrong.
$$
e_{ensemble} = \sum_{i=13}^{25} 
\begin{pmatrix}
25 \\ i
\end{pmatrix}
e^i(1-e)^{25-i} = 0.06
$$
[..]

In general, ensemble methods are useful if:
1. the base classifiers are independent
2. the performance of the base classifier is better than random choice

With a manipulation of the training set we can obtain some independence. 
There are 2 major methods for dataset manipulation:
- __Bagging__ repeatedly samples with replacement (in the training set) according to a uniform probability distribution. 
- Boosting iteratively changes the distribution of training examples so that the base classifier focus on examples which are hard to classify
- Adaboost the importance of each base classifier depends on its error rate

