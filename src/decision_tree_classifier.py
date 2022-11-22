"""Scrach implementation for a binary decision tree classifier."""

from collections import Counter
import numpy as np
from typing import Union, Dict


class Node:
    """
    _summary_
    """

    def __init__(
        self, mean=None, depth=None, column=None, value=None, gini=None, samples=None, left=None, right=None
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        mean : _type_, optional
            _description_, by default None
        depth : _type_, optional
            _description_, by default None
        column : _type_, optional
            _description_, by default None
        value : _type_, optional
            _description_, by default None
        gini : _type_, optional
            _description_, by default None
        samples : _type_, optional
            _description_, by default None
        left : _type_, optional
            _description_, by default None
        right : _type_, optional
            _description_, by default None
        """
        self.mean = mean
        self.depth = depth
        self.column = column
        self.value = value
        self.gini = gini
        self.samples = samples
        self.left = left
        self.right = right


class DecisionTreeClassifier:
    """
    _summary_
    """

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        max_depth : int, optional
            _description_, by default None
        min_samples_split : int, optional
            _description_, by default 2
        min_samples_leaf : int, optional
            _description_, by default 1
        min_impurity_decrease : float, optional
            _description_, by default 0.0
        """
        self.tree = None
        self.max_depth = np.inf if max_depth == None else max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurit_decrease = min_impurity_decrease

    @staticmethod
    def _compute_gini(n0, n1) -> float:
        """
        _summary_

        Parameters
        ----------
        n0 : _type_
            _description_
        n1 : _type_
            _description_

        Returns
        -------
        float
            _description_
        """
        if n0 == 0 or n1 == 0:
            return 0

        n = n0 + n1

        # Scale counts.
        p0 = n0 / n
        p1 = n1 / n

        # Compute gini impurity.
        gini = 1 - (p0**2 + p1**2)

        return gini

    def _search_best_split(self, X: np.array, y: np.array) -> Dict[str, Union[float, int]]:
        """
        Return the column and value on where to split dataset to produce the larger information gain.

        Parameters
        ----------
        X : np.array
            Array-like containing features as columns and samples as rows.
        y : np.array
            Array-like containing target variable.

        Returns
        -------
        Dict[str, Union[float, int]]
            Dictionary containing column and value on where to perform the split and gini obtained.
        """
        best_col = None
        best_value = None
        # Count 0/1's.
        counter = Counter(y)
        n = len(X)
        parent_gini = self._compute_gini(counter[0], counter[1])
        best_gini = parent_gini

        if n >= self.min_samples_split:

            # Iterate through every column.
            for col in range(X.shape[1]):
                # Sort X and y based on col.
                ordered_index = np.argsort(X[:, col])
                X = X[ordered_index]
                y = y[ordered_index]

                # Start assigning all samples into left branch.
                left_n0, left_n1 = 0, 0
                right_n0, right_n1 = counter[0], counter[1]

                for row, value in enumerate(X[:, col]):
                    target = y[row]

                    # Update left branch by substracting target value.
                    left_n0 += 1 - target
                    left_n1 += target
                    left_n = left_n0 + left_n1
                    left_gini = self._compute_gini(left_n0, left_n1) * (left_n)

                    # Update right branch by adding target value.
                    right_n0 -= 1 - target
                    right_n1 -= target
                    right_n = right_n0 + right_n1
                    right_gini = self._compute_gini(right_n0, right_n1) * (right_n)

                    # Compute weighted gini.
                    gini = (left_gini + right_gini) / n

                    # If impurity decreases, then save split.
                    if (
                        best_gini - gini > self.min_impurit_decrease
                        and left_n >= self.min_samples_leaf
                        and right_n >= self.min_samples_leaf
                    ):
                        best_col = col
                        best_value = value
                        best_gini = gini

        return (best_col, best_value, parent_gini, n)

    def fit(self, X: np.array, y: np.array, depth=0) -> None:
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : np.array
            Array-like containing features as columns and samples as rows.
        y : np.array
            Array-like containing target variable.
        depth : int, optional
            Indicate depth of current node, by default 0. Don't need to be called. It's used for the recursion.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        column, value, gini, samples = self._search_best_split(X, y)
        if self.tree == None:
            self.tree = node = Node(
                mean=np.mean(y), depth=depth, column=column, value=value, gini=gini, samples=samples
            )
        else:
            node = Node(mean=np.mean(y), depth=depth, column=column, value=value, gini=gini, samples=samples)

        if node.column and node.value and depth <= self.max_depth:

            left_condition = X[:, node.column] <= node.value
            right_condition = ~left_condition

            node.left = self.fit(X[left_condition], y[left_condition], depth + 1)
            node.right = self.fit(X[right_condition], y[right_condition], depth + 1)
        return node

    def predict_proba(self, X: np.array) -> np.array:
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : np.array
            Array-like containing features as columns and samples as rows.

        Returns
        -------
        np.array
            1th dimensional array containing probability of sample to be equal 1.
        """
        # Initialize a 1 dimensional array for predictions.
        y = np.zeros(len(X))

        # Traverse the tree and take the mean of the correspondent leaf.
        for sample in range(len(X)):
            node = self.tree
            while node.left and node.right:
                if X[sample, node.column] <= node.value:
                    node = node.left
                else:
                    node = node.right
            # Store prediction.
            y[sample] = node.mean
        return y
    
    def predict(self, X: np.array) -> np.array:
        """Predict class or regression value for X.
        
        Parameters
        ----------
        X : np.array
            Array-like containing features as columns and samples as rows.

        Returns
        -------
        np.array
            1th dimensional array containing predicted classes.
        """
        return np.round(self.predict_proba(X))
