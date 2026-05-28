"""Tests for model training pipeline."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from ml_arch_recommender.model.train import build_pipeline, generate_synthetic_data, train


def test_synthetic_data_shape():
    X, y = generate_synthetic_data(n_samples=200)
    assert X.shape == (200, 5)
    assert y.shape == (200,)


def test_synthetic_data_has_both_classes():
    _, y = generate_synthetic_data(n_samples=500)
    assert 0 in y and 1 in y


def test_pipeline_trains_and_predicts():
    X, y = generate_synthetic_data(n_samples=300)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X[:5])
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})


def test_pipeline_predict_proba():
    X, y = generate_synthetic_data(n_samples=300)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_train_saves_model_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        metadata = train(output_dir)
        assert (output_dir / "credit_model.joblib").exists()
        assert (output_dir / "model_metadata.json").exists()
        assert "auc_roc" in metadata["metrics"]
        assert metadata["metrics"]["auc_roc"] > 0.7
