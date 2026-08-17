"""
evaluate.py — Model değerlendirme modülü.

Kaydedilmiş en iyi modeli test verisi üzerinde değerlendirir.
Accuracy, precision, recall, F1-score, classification report ve
confusion matrix hesaplayıp kaydeder.
"""

import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import config
import utils
from model import load_trained_model


# =============================================================================
# Tahmin Toplama
# =============================================================================

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = config.DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Model ile tüm batch'lerdeki tahminleri toplar.

    Args:
        model: Değerlendirilecek model.
        dataloader: Test DataLoader'ı.
        device: Hesaplama cihazı.

    Returns:
        (all_labels, all_predictions) numpy array'leri.
    """
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# =============================================================================
# Metrikleri Hesaplama
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
) -> Dict:
    """
    Tüm değerlendirme metriklerini hesaplar.

    Args:
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.
        class_names: Sınıf isimleri.

    Returns:
        Metrikleri içeren sözlük.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Her sınıf için ayrı precision, recall, F1
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )

    # Classification report (metin)
    cls_report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "classification_report": cls_report,
        "confusion_matrix": cm,
    }

    return metrics


# =============================================================================
# Sonuçları Yazdırma
# =============================================================================

def print_metrics(metrics: Dict, class_names: List[str] = None) -> None:
    """
    Hesaplanan metrikleri düzenli biçimde ekrana yazdırır.

    Args:
        metrics: compute_metrics çıktısı.
        class_names: Sınıf isimleri.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    print("\n" + "=" * 60)
    print("  TEST SETİ DEĞERLENDİRME SONUÇLARI")
    print("=" * 60)

    print(f"\n  Accuracy         : {metrics['accuracy']:.4f} "
          f"({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro)   : {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro) : {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Wt.)   : {metrics['f1_weighted']:.4f}")

    print("\n  Sınıf Bazında Sonuçlar:")
    print(f"  {'Sınıf':15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print("  " + "-" * 47)
    for i, cls_name in enumerate(class_names):
        print(f"  {cls_name:15s} "
              f"{metrics['precision_per_class'][i]:10.4f} "
              f"{metrics['recall_per_class'][i]:10.4f} "
              f"{metrics['f1_per_class'][i]:10.4f}")

    print(f"\n  Classification Report:\n{metrics['classification_report']}")

    print("  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    # Başlık satırı
    header = "  " + " " * 15 + "".join(f"{c:>12s}" for c in class_names)
    print(header)
    for i, cls_name in enumerate(class_names):
        row = "  " + f"{cls_name:15s}" + "".join(f"{cm[i, j]:12d}" for j in range(len(class_names)))
        print(row)

    print("=" * 60)


# =============================================================================
# Sonuçları Dosyaya Kaydetme
# =============================================================================

def save_metrics(metrics: Dict,
                  class_names: List[str] = None,
                  reports_dir: str = config.REPORTS_DIR) -> None:
    """
    Metrikleri metin ve CSV formatında kaydeder.

    Args:
        metrics: compute_metrics çıktısı.
        class_names: Sınıf isimleri.
        reports_dir: Raporların kaydedileceği dizin.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    os.makedirs(reports_dir, exist_ok=True)

    # --- Metin raporu ---
    txt_path = os.path.join(reports_dir, "evaluation_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Beyin Tumoru MRI Siniflandirma -- Test Sonuclari\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Accuracy         : {metrics['accuracy']:.4f} "
                f"({metrics['accuracy'] * 100:.2f}%)\n")
        f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro)   : {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro) : {metrics['f1_macro']:.4f}\n")
        f.write(f"F1-Score (Wt.)   : {metrics['f1_weighted']:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(metrics["classification_report"])
        f.write("\n\nConfusion Matrix:\n")

        # Confusion matrix tablosu
        cm = metrics["confusion_matrix"]
        header = " " * 15 + "".join(f"{c:>12s}" for c in class_names) + "\n"
        f.write(header)
        for i, cls_name in enumerate(class_names):
            row = f"{cls_name:15s}" + "".join(
                f"{cm[i, j]:12d}" for j in range(len(class_names))
            ) + "\n"
            f.write(row)

    print(f"[SAVE] Metin raporu kaydedildi: {txt_path}")

    # --- Genel metrikler CSV'si ---
    summary_path = os.path.join(reports_dir, "evaluation_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metrik", "Değer"])
        writer.writerow(["Accuracy", f"{metrics['accuracy']:.4f}"])
        writer.writerow(["Precision (Macro)", f"{metrics['precision_macro']:.4f}"])
        writer.writerow(["Recall (Macro)", f"{metrics['recall_macro']:.4f}"])
        writer.writerow(["F1-Score (Macro)", f"{metrics['f1_macro']:.4f}"])
        writer.writerow(["F1-Score (Weighted)", f"{metrics['f1_weighted']:.4f}"])

    print(f"[SAVE] Özet CSV kaydedildi: {summary_path}")

    # --- Sınıf bazında metrikler CSV'si ---
    per_class_path = os.path.join(reports_dir, "evaluation_per_class.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sınıf", "Precision", "Recall", "F1-Score"])
        for i, cls_name in enumerate(class_names):
            writer.writerow([
                cls_name,
                f"{metrics['precision_per_class'][i]:.4f}",
                f"{metrics['recall_per_class'][i]:.4f}",
                f"{metrics['f1_per_class'][i]:.4f}",
            ])

    print(f"[SAVE] Sınıf bazlı CSV kaydedildi: {per_class_path}")


# =============================================================================
# Ana Değerlendirme Fonksiyonu
# =============================================================================

def evaluate_model(
    test_loader: DataLoader,
    model_path: str = config.BEST_MODEL_PATH,
    device: torch.device = config.DEVICE,
    class_names: List[str] = None,
) -> Dict:
    """
    Kaydedilmiş en iyi modeli test seti üzerinde değerlendirir.

    Args:
        test_loader: Test DataLoader'ı.
        model_path: Kaydedilmiş model dosya yolu.
        device: Hesaplama cihazı.
        class_names: Sınıf isimleri.

    Returns:
        Metrikleri içeren sözlük.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    print("\n[EVAL] Test seti değerlendirmesi başlıyor...")

    # Modeli yükle
    model = load_trained_model(model_path, device=device)

    # Tahminleri topla
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # Metrikleri hesapla
    metrics = compute_metrics(y_true, y_pred, class_names)

    # Sonuçları yazdır
    print_metrics(metrics, class_names)

    # Sonuçları kaydet
    save_metrics(metrics, class_names)

    # Confusion matrix görseli oluştur
    utils.plot_confusion_matrix(metrics["confusion_matrix"], class_names)

    return metrics
