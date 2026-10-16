import numpy as np

from revision.budget_learning import select_thresholds, fit_operation, AbstainingClassifier


def test_missing_positive_support_abstains_without_resampling_labels():
    model, counts = fit_operation(np.zeros((8, 28)), np.zeros(8))
    assert isinstance(model, AbstainingClassifier)
    assert not model.predict_proba(np.zeros((2, 28)))[:, 1].any()
    thresholds, details = select_thresholds(np.array([1., 0.]), np.array([.99, .01]), np.array([0, 0]))
    assert thresholds == [1.1, 1.1, 1.1]


def test_threshold_uses_declared_precision_and_minimum_correct():
    y = np.array([1.] * 5 + [0.] * 5)
    probability = np.array([.8] * 5 + [.2] * 5)
    thresholds, details = select_thresholds(y, probability, np.zeros(10, dtype=int))
    assert thresholds == [.8, 1.1, 1.1]
    assert details[0]['precision'] == 1.
    assert details[0]['correct_actions'] == 5
