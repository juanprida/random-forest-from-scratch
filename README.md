## Scratch implementation for tree based models.
 
<p align="center">
  <img src="https://github.com/juanprida/tree-based-models-from-scratch/blob/master/tree_picture.jpg?raw=true" alt="Tree image"/>
</p>

A minimal implementation of different tree-based-models with a scikit learn-like API (`.fit()` and `.predict()` ;)).

First model (and only so far) is `DecisionTreeClassifier` which looks to minimize impurity in target data by creating partitions of the samples. We stick to gini coefficient for measuring impurity.

### Example usage
```
from decision_tree_classifier import DecisionTreeClassifier

# Initialize class. 
decision_tree_model = DecisionTreeClassifier(
        max_depth = 10,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_impurity_decrease = 0.0,
)

# Fit.
decision_tree_model.fit(X, y)

# Predict.
y_train_hat = model.predict(X)
```

### Project structure.
There's an execution of the models in `demo.ipynb`.

Inside of `src.decision_tree_classifier.py` you can find all the code needed to create the DecisionTreeClassifier.

#### Main steps are:
- Compute gini coefficients with `_compute_gini()`.
- Find best split for a given node with `_search_best_split()`
- Grow the full tree using recursion with (obviosuly) `fit()`
- Traverse the tree with `predict()`.
