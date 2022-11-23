"""Scratch implementation of a decision tree classifier."""

from collections import Counter
from typing import Union, Tuple
import numpy as np


class Node:
    """Base class for tree nodes."""

    def __init__(
        self,
        mean: float,
        depth: int,
        column: int,
        value: float,
        gini: float,
        samples: int,
        left=None,
        right=None,
    ) -> None:
        """
        Init method.

        Parameters
        ----------
        mean : float
            Indicate constant prediction for the node.
        depth : int
            Indicate depth of current node in the tree.
        column : int
            Indicate colum on where to perform split for children branches.
        value : float
            Indicate value on where to perform the split for children branches.
        gini : float
            Indicate gini impurity at the node.
        samples : int
            Indicate
        left : Node, optional
            Hold left children node, by default None.
        right : Node, optional
            Hold left children node, by default None.
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
    """Custom decision tree classifier."""

    def __init__(
        self,
        max_depth: int = np.inf,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """
        Init method.

        Parameters
        ----------
        max_depth : int, np.inf
            Indicate maximum depth of the tree, by default np.inf
        min_samples_split : int, optional
            Indicate minimum number of samples to split a node, by default 2
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node, by default 1
        min_impurity_decrease : float, optional
            A split will be performed decrease of the impurity is greater than this value, by default 0.0
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurit_decrease = min_impurity_decrease

    @staticmethod
    def _compute_gini(negative_samples: int, positive_samples: int) -> float:
        """
        Measure gini impurity for binary distribution.

        Parameters
        ----------
        negative_samples : int
            Indicate amount of negative (0s) samples.
        positive_samples : int
            Indicate amount of positve (1s) samples.

        Returns
        -------
        float
            Gini coefficient. Values will be between 0 and 0.5.
        """
        if negative_samples == 0 or positive_samples == 0:
            return 0.0

        # Scale counts.
        samples = negative_samples + positive_samples
        p0 = negative_samples / samples
        p1 = positive_samples / samples

        # Compute gini impurity.
        gini = 1.0 - (p0**2 + p1**2)

        return gini

    def _search_best_split(self, X: np.array, y: np.array) -> Tuple[Union[float, int]]:
        """
        Return the column and value on where to split dataset to produce the larger impurity decrease.

        Parameters
        ----------
        X : np.array
            Array-like containing features as columns and samples as rows.
        y : np.array
            Array-like containing target variable.

        Returns
        -------
        Tuple[Union[float, int]]
            Tuple containing column and value on where to perform the split, gini obtained and samples in the node.
        """
        best_col = None
        best_value = None
        # Count negative and positive samples.
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

                # Start assigning all samples into right branch.
                left_n0, left_n1 = 0, 0
                right_n0, right_n1 = counter[0], counter[1]

                # For every row, we update left and right branch with y values.
                for row, value in enumerate(X[:, col]):
                    target = y[row]

                    # Update left branch by adding target value.
                    left_n0 += 1 - target
                    left_n1 += target
                    left_n = left_n0 + left_n1
                    left_gini = self._compute_gini(left_n0, left_n1) * (left_n)

                    # Update right branch by substracting target value.
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

    def fit(self, X: np.array, y: np.array, depth=0) -> Node:
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
        # Search best split and initialize a node with the information.
        column, value, gini, samples = self._search_best_split(X, y)
        node = Node(mean=np.mean(y), depth=depth, column=column, value=value, gini=gini, samples=samples)

        # Set first node as root.
        if self.tree is None:
            self.tree = node

        # If there's a possible split to do, repeat the process.
        if node.column is not None and node.value is not None and depth <= self.max_depth:
            # Define conditions for both branches.
            left_condition = X[:, node.column] <= node.value
            right_condition = ~left_condition
            # Let the magic begin.
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
        return np.round(self.predict_proba(X)).astype('int')
