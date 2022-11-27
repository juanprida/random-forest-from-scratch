"""Scratch implementation of a Random Forest Classifier."""
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from src.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest Classifier.

    The class is built upon DecisionTreeClassifier. It is an ensemble method that constructs multiple decision trees
    applying bootstrapping and takes the average between all the trees.

    Attributes
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Indicate maximum depth of the tree.
    min_samples_split : int
        Indicate minimum number of samples to split a node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    max_features : int
        Number of features to keep for every tree in the forest.
        If None, all features are used.
    min_impurity_decrease : float
        A split will be performed decrease of the impurity is greater than this value.
    bootstrap : bool
        Wheter to bootstrap samples when building the tree. If False, whole dataset is used to build each tree.
    n_cores : int
        Number of cores to use when parallelizing fit loop.
    estimators : List[DecisionTreeClassifier]
        Collection of fitted decision trees classifiers.

    References
    ----------
    [1] Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = np.inf,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        n_cores: int = None,
    ) -> None:
        """
        Init method.

        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the forest, by default 100
        max_depth : int, optional
            Indicate maximum depth of the tree, by default np.inf
        min_samples_split : int, optional
            Indicate minimum number of samples to split a node, by default 2
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node, by default 1
        min_impurity_decrease : float, optional
            A split will be performed decrease of the impurity is greater than this value, by default 0.0
        bootstrap : bool, optional
            Wheter to bootstrap samples when building the tree, by default True
        n_cores : int, optional
            Number of cores to use when parallelizing fit loop, by default None
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.n_cores = multiprocessing.cpu_count() if n_cores is None else n_cores
        decision_tree_params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
        }
        self.estimators = [DecisionTreeClassifier(**decision_tree_params) for n in range(self.n_estimators)]

    def _fit_bootstrapped(self, estimator: int, X: np.array, y: np.array) -> None:
        """
        Build a bootstrapped decision tree classifier from the training set (X, y).

        Parameters
        ----------
        estimator_index : int
            Indicate index of the tree to be fitted.
        X : np.array
            Array-like containing features as columns and samples as rows.
        y : np.array
            Array-like containing target variable.
        """
        index = [i for i in range(X.shape[0])]
        X_tree = X.copy()
        y_tree = y.copy()

        if self.bootstrap:
            index = np.random.choice(index, X.shape[0], replace=True)
            X_tree = X_tree[index]
            y_tree = y_tree[index]

        self.estimators[estimator].fit(X_tree, y_tree)

        return self.estimators[estimator]

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : np.array
            Array-like containing features as columns and samples as rows.
        y : np.array
            Array-like containing target variable.
        """
        range_forest = range(self.n_estimators)
        self.estimators = Parallel(n_jobs=self.n_cores)(
            delayed(self._fit_bootstrapped)(estimator, X, y) for estimator in range_forest
        )

    def predict_proba(self, X: np.array) -> np.array:
        """Predict probabilities of the input samples X.

        Parameters
        ----------
        X : np.array
            Array-like containing features as columns and samples as rows.

        Returns
        -------
        np.array
            1th dimensional array containing probability of sample to be equal 1.
        """
        probas = []
        for tree in self.estimators:
            y_pred_tree = tree.predict_proba(X)
            probas.append(y_pred_tree)

        return np.mean(probas, axis=0)

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
        return np.round(self.predict_proba(X)).astype("int")
