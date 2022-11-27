## Scratch implementation for different tree based models.
A minimal implementation of different tree-based-models with a scikit learn-like API (`.fit()` and `.predict()` ;)).

### Introduction.
Tree based models are the most popular types of models used in machine learning for tabular data mostly because of their high accuracy and relatively low computational cost.

- Decision trees are a type of supervised learning algorithm that are used for both classification and regression tasks. The algorithm works by splitting the data into smaller subsets, and then making predictions based on the data in those subsets.
Even if decision trees are a good first approach offering high interpretability, they are are prone to overfitting the training data. Therefore, its use in practice today is insignificant. However, they remain as a base for more poweful techniques like Random Forest or Gradient Boosting Machines.

- Random Forest is an ensemble learning method which works by training multiple decision trees on different subsets of the data by resampling the original dataset with replacement, and then averaging the predictions of those trees. They are less likely to overfit than a single decision tree, and can make accurate predictions on new data. However, they are more computationally expensive than a single decision tree (as an example scikit-learn Random Forest by default averages 100 decision trees).

- Finally, Gradient Boosting Machines work by training multiple one or two level decision trees on on top of each other, and then combining the predictions of those trees. They are also less likely to overfit than a single decision tree, and tend to be more accurate (in general) than Random Forest in practice.

In this project, we build from scratch two classifiers: `DecisionTreeClassifier` and `RandomForestClassifier`.

### Decision Tree Classifier

<p align="center">
  <img src="https://github.com/juanprida/tree-based-models-from-scratch/blob/master/tree_picture.jpg?raw=true" alt="Tree image"/>
</p>

Inside of `src.decision_tree_classifier.py` you can find all the code needed to create the `DecisionTreeClassifier`.

#### Main steps are:
- **Compute impurity:**
</br> The core idea of a Decision Tree lies behind the concept of impurity. We say that a distribution is pure if its elements are similar between them.
</br> For example, suppose that we are working with a binary dataset. If a large proportion of the samples belongs to the same class (either 0 or 1) then we can say that the impurity in our data is quite low.
</br> Luckily for us, there are a good amount of ways to translate this idea into plain algebra. Here we stick with Gini impurity. So, just remember the impurity of a dataset tries to explain how homogeneous it is.
</br> Check the `_compute_gini()` for the implementation.

- **Find best split for a given node:**
Another useful term is **Information gain** and it references to how much can we reduce the impurity by performing a split to a given node by a given feature and value. We store the average of the target values since we will be using them for our predictions.
</br> This step is covered under `_search_best_split()` where we evaluate all the possible splits (looking at every possible partition) and return which is the optimal split for a given dataset.

- **Grow full Decision Tree:**
Computing impurity and find best split, accounts for 90% of the job. The rest is quite straightforward, we use recursion and repeat the process for every partition until we don't experience a decrease in impurity by performing a split.
</br> The code is implemented under the method `.fit()`.

- Traverse the tree with `predict()`.
When we need to return a prediction, we traverse the fitted tree with a new input and return the mean of the leaf node.

#### Example usage
```
from decision_tree_classifier import DecisionTreeClassifier

# Initialize class. 
decision_tree = DecisionTreeClassifier(
        max_depth = 10,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_impurity_decrease = 0.0,
)

# Fit.
decision_tree.fit(X, y)

# Predict.
y_train_hat = decision_tree.predict(X)
```

### Random Forest Classifier

<p align="center">
  <img src="https://github.com/juanprida/tree-based-models-from-scratch/blob/master/forest_picture.jpg?raw=true" alt="Tree image"/>
</p>

Inside of `src.random_forest_classifier.py` you can find all the code needed to create the `RandomForestClassifier`.

