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
</br> The core idea of a Decision Tree lies behind the concept of impurity. We say that a distribution is "more pure" if its elements are similar between them. 
</br> For example, suppose that we are working with a binary variable. If a large proportion of the samples belongs to the same class (either 0 or 1) then we can say that the impurity in our data is low.
</br> Luckily for us, there are a good amount of ways to translate this idea into plain algebra. Here we stick to Gini impurity.
</br> Check the `_compute_gini()` for the implementation.

- **Find best split for a given node:**
</br> Another useful term is **Information gain** and it refers to how much we can reduce the impurity by performing a split to a given node by a certain feature and value.
</br> In `_search_best_split()` we evaluate all the possible splits (looking at every possible partition) and return which is the optimal split for a given node.

- **Grow full Decision Tree:**
</br> The main idea behind decision trees is that we start with a root node for an incoming dataset. From there, we perform splits looking to reduce the impurity. At every partition, we store the average value of the target variable which will be used as the model predictions.
</br> In reality, computing impurity and find the best split, accounts for 90% of the job. The rest is quite straightforward, we use recursion and repeat the process for every partition until we don't experience a decrease in impurity by performing a split.
</br> The code is implemented under the method `fit()`.

- **Traverse the tree and get predictions:**
</br> When we need to return a prediction, we traverse the fitted tree with a new input and return the mean of the leaf node.
</br> The code is implemented under the method `predict_proba()` and `predict()`.

#### Example usage
```
from src.decision_tree_classifier import DecisionTreeClassifier

# Initialize class. 
decision_tree = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease = 0.0,
)

# Fit.
decision_tree.fit(X, y)

# Predict.
y_hat = decision_tree.predict(X)
```

### Random Forest Classifier

<p align="center">
  <img src="https://github.com/juanprida/tree-based-models-from-scratch/blob/master/forest_picture.jpg?raw=true" alt="Tree image"/>
</p>

Inside of `src.random_forest_classifier.py` you can find all the code needed to create the `RandomForestClassifier`.

#### Main steps are:
- **Initialize *n* `DecisionTreeClassifier`s:**
</br> Very first step is to create a Forest .
</br> Inside `__init__()`, we initilize multiple instances of `DecisionTreeClassifier`s under `estimators`.

- **Fit one tree with bootstrapping:**
</br> The beauty of Random Forest can be attibuted to bootstrapping which is a sampling technique used to create multiple datasets from the original dataset. This allows for the creation of multiple decision trees, each of which is trained on a different dataset.
</br> See `_fit_bootstrapped()` for the implementation.

- **Fit multiples trees in parallel using `joblib`:**
</br> We could have a simple `for` loop fitting every tree on the forest. However, since all the trees are independent from each other, we can leverage `joblib` in order to run the different trees in parallel.
</br> We do this in `fit()` (don't get confused, this is a different fit that the one that we've seen in `DecisionTree`!).

- **Get predictions:**
</br> Once we have multiple trees fitted, in `predict_proba` we just need to take the average of all the predictions.

#### Example usage
```
from src.random_forest_classifier import RandomForestClassifier

# Initialize class. 
random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=8,
    min_samples_leaf=1,
    min_impurity_decrease=0.0,
    bootstrap=True,
    n_cores=None,
)

# Fit.
random_forest.fit(X, y)

# Predict.
y_hat = random_forest.predict(X)
```

