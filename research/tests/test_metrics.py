import unittest

from intraop.evaluation.metrics import binary_classification_metrics


class MetricTests(unittest.TestCase):
    def metrics(self, labels, probabilities, **overrides):
        options = {"positive_label": "b", "negative_label": "a", "threshold": 0.6}
        return binary_classification_metrics(labels, probabilities, **{**options, **overrides})

    def test_explicit_class_mapping_and_threshold(self):
        result = self.metrics(["a", "b", "a", "b"], [0.1, 0.9, 0.4, 0.6])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["auroc"], 1)
        self.assertEqual(result["average_precision"], 1)
        self.assertAlmostEqual(result["brier_score"], 0.085)

    def test_single_class_metrics_are_explicitly_undefined(self):
        result = self.metrics(["a", "a"], [0.1, 0.2])
        for metric in ("auroc", "average_precision", "recall", "precision"):
            self.assertIsNone(result[metric])
        self.assertEqual(result["accuracy"], 1)
        result = self.metrics(["b", "b"], [0.8, 0.9])
        self.assertIsNone(result["auroc"])
        self.assertEqual(result["recall"], 1)

    def test_invalid_inputs_rejected(self):
        cases = [
            (["a"], [1.1], {}),
            (["a"], [float("nan")], {}),
            (["a"], [], {}),
            (["c"], [0.5], {}),
            ([], [], {}),
            (["a"], [0.5], {"threshold": -1}),
            (["a"], [0.5], {"threshold": float("nan")}),
            (["a"], [0.5], {"positive_label": "a"}),
        ]
        for labels, probabilities, overrides in cases:
            with self.subTest(), self.assertRaises(ValueError):
                self.metrics(labels, probabilities, **overrides)
