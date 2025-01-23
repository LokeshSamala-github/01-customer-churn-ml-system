import json

from churn_system.config import PATHS


def test_training_produces_model_and_metrics(trained_model):
    assert PATHS.model_file.exists()
    assert PATHS.metrics_file.exists()
    assert PATHS.feature_manifest.exists()


def test_metrics_report_shape(trained_model):
    report = json.loads(PATHS.metrics_file.read_text())
    for split in ("train", "validation", "test"):
        assert split in report["metrics"]
        m = report["metrics"][split]
        assert 0.5 <= m["roc_auc"] <= 1.0, "a trained model should beat random guessing on every split"
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0


def test_test_set_auc_is_believable_not_suspicious(trained_model):
    """A portfolio red flag is a model claiming ~0.99+ AUC on synthetic data
    -- it means the label leaked into a feature. This dataset has
    irreducible noise baked into the generator on purpose; assert the
    trained model's test AUC stays in a believable band."""
    report = json.loads(PATHS.metrics_file.read_text())
    test_auc = report["metrics"]["test"]["roc_auc"]
    assert 0.65 < test_auc < 0.95


def test_search_trials_were_recorded(trained_model):
    report = json.loads(PATHS.metrics_file.read_text())
    assert len(report["search_trials"]) == 3  # matches --search-iters 3 in the fixture
    aucs = [t["val_auc"] for t in report["search_trials"]]
    assert aucs == sorted(aucs, reverse=True), "trials should be sorted best-first"
