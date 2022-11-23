"""Tests for Decision Tree Classifier."""
import numpy as np
from src.decision_tree_classifier import DecisionTreeClassifier


def test_compute_gini() -> None:

    dtc = DecisionTreeClassifier()
    assert dtc._compute_gini(10, 0) == 0

    assert dtc._compute_gini(10, 10) == 0.50


def test_sanity_check():

    X = np.array(
        (
            [
                [-0.22282357, -0.67769005, -0.20563921, -0.72526148],
                [-0.27640402, -0.71059994, -0.2020569, -0.56249352],
                [-0.27036556, -0.05127691, 0.06488306, 1.11891756],
                [0.49572314, 1.01022237, 0.25464126, 0.32379779],
                [-1.08812049, -1.60046208, -0.30734696, 0.88888302],
                [0.76061912, 0.85308434, 0.1065072, -1.31013392],
                [-0.18580931, 0.27921977, 0.17282023, 1.58425194],
                [-0.71578272, -1.06147299, -0.20571059, 0.56225893],
                [0.9801154, 2.2983572, 0.62620405, 1.42058024],
                [-0.42970093, -1.35176973, -0.41486627, -1.51499752],
            ]
        )
    )

    y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 0])

    dtc = DecisionTreeClassifier()
    dtc.fit(X, y)
    assert (dtc.predict(X) == y).all()
