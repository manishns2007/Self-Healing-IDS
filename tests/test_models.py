"""
Tests for ML models — feature engineering, supervised, anomaly, and ensemble.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.preprocessing.feature_engineering import FeatureEngineer
from src.models.supervised import RandomForestDetector, XGBoostDetector
from src.models.anomaly import IsolationForestDetector
from src.models.ensemble import EnsembleDetector
from src.ingestion.data_loader import load_or_generate_synthetic


@pytest.fixture(scope="module")
def sample_data():
    df = load_or_generate_synthetic(n_samples=2000)
    fe = FeatureEngineer()
    X = fe.fit_transform(df)
    y = df["is_attack"].values
    return X, y, fe


def test_feature_engineer(sample_data):
    X, y, fe = sample_data
    assert X.shape[0] == 2000
    assert X.shape[1] > 38  # At least NSL-KDD cols + engineered features
    assert fe.n_features == X.shape[1]
    assert not np.isnan(X).any()


def test_random_forest(sample_data):
    X, y, fe = sample_data
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]

    rf = RandomForestDetector()
    rf.fit(X_train, y_train)
    assert rf.is_fitted

    preds = rf.predict(X_test)
    assert preds.shape == (500,)
    assert set(preds).issubset({0, 1})

    probs = rf.attack_probability(X_test)
    assert probs.shape == (500,)
    assert (probs >= 0).all() and (probs <= 1).all()

    metrics = rf.evaluate(X_test, y_test)
    assert metrics["f1"] > 0.5  # Reasonable baseline


def test_xgboost(sample_data):
    X, y, _ = sample_data
    xgb = XGBoostDetector()
    xgb.fit(X[:1500], y[:1500])
    assert xgb.is_fitted
    preds = xgb.predict(X[1500:])
    assert set(preds).issubset({0, 1})


def test_isolation_forest(sample_data):
    X, y, _ = sample_data
    iso = IsolationForestDetector()
    iso.fit(X[:1500], y[:1500])
    assert iso.is_fitted
    scores = iso.anomaly_score(X[1500:])
    assert (scores >= 0).all() and (scores <= 1).all()


def test_ensemble(sample_data):
    X, y, _ = sample_data
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]

    ens = EnsembleDetector()
    ens.fit(X_train, y_train)
    assert ens.is_fitted

    preds, scores = ens.predict_with_score(X_test)
    assert preds.shape == (500,)
    assert (scores >= 0).all() and (scores <= 1).all()

    metrics = ens.evaluate(X_test, y_test)
    assert metrics["f1"] > 0.5
    assert metrics["roc_auc"] > 0.7

    # Single prediction
    single = ens.predict_single(X_test[0:1])
    assert isinstance(single["is_attack"], bool)
    assert "model_scores" in single
    assert "ensemble_score" in single
