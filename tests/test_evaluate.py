"""tests/test_evaluate.py — evaluate.py testleri.

Buradaki test_save_metrics_writes_two_readable_csvs, Model Performansi
sayfasindaki CSV okuma hatasi icin bir regresyon testidir: save_metrics
eskiden tek, duzensiz (ragged) bir CSV yaziyordu ve pd.read_csv bunu
ayristiramiyordu. Bu test, artik iki ayri, gecerli CSV yazildigini ve
pandas'in ikisini de sorunsuz okuyabildigini dogrular.
"""

import numpy as np
import pandas as pd

import config
from evaluate import compute_metrics, save_metrics


def _fake_metrics():
    # 4 sinif icin basit, gercekci sahte tahmin/gercek etiketler
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 3, 2])
    return compute_metrics(y_true, y_pred, class_names=config.CLASS_NAMES)


def test_compute_metrics_shape():
    metrics = _fake_metrics()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["precision_per_class"]) == config.NUM_CLASSES
    assert metrics["confusion_matrix"].shape == (config.NUM_CLASSES, config.NUM_CLASSES)


def test_save_metrics_writes_two_readable_csvs(tmp_path):
    metrics = _fake_metrics()

    save_metrics(metrics, class_names=config.CLASS_NAMES, reports_dir=str(tmp_path))

    summary_path = tmp_path / "evaluation_summary.csv"
    per_class_path = tmp_path / "evaluation_per_class.csv"
    assert summary_path.exists()
    assert per_class_path.exists()

    # pd.read_csv hic exception firlatmadan okuyabilmeli -- asil regresyon kontrolu bu
    summary_df = pd.read_csv(summary_path)
    class_df = pd.read_csv(per_class_path)

    assert list(summary_df.columns) == ["Metrik", "Değer"]
    assert len(summary_df) == 5

    assert list(class_df.columns) == ["Sınıf", "Precision", "Recall", "F1-Score"]
    assert len(class_df) == config.NUM_CLASSES
    assert set(class_df["Sınıf"]) == set(config.CLASS_NAMES)


def test_save_metrics_no_longer_writes_ragged_combined_csv(tmp_path):
    metrics = _fake_metrics()
    save_metrics(metrics, class_names=config.CLASS_NAMES, reports_dir=str(tmp_path))

    assert not (tmp_path / "evaluation_results.csv").exists()
