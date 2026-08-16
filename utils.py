"""
utils.py — Ortak yardımcı fonksiyonlar.

Random seed ayarlama, grafik çizme, dizin oluşturma ve diğer
paylaşılan işlevleri içerir.
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlar için (sunucu, CI vb.)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

import config


# =============================================================================
# Tekrar Üretilebilirlik
# =============================================================================

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """
    Tüm random seed değerlerini ayarlar.
    Python, NumPy, PyTorch (CPU + CUDA) için tekrar üretilebilirlik sağlar.

    Args:
        seed: Kullanılacak seed değeri.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU durumunda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[SEED] Random seed ayarlandı: {seed}")


# =============================================================================
# Dizin Oluşturma
# =============================================================================

def ensure_dirs() -> None:
    """
    Proje için gerekli çıktı dizinlerini oluşturur.
    Zaten mevcutsa hata vermez.
    """
    dirs = [config.MODELS_DIR, config.FIGURES_DIR, config.REPORTS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[DIRS] Çıktı dizinleri hazır.")


# =============================================================================
# Grafik Çizme Fonksiyonları
# =============================================================================

def plot_training_history(history: Dict[str, List[float]],
                          save_dir: str = config.FIGURES_DIR) -> None:
    """
    Eğitim geçmişinden loss ve accuracy grafiklerini çizer.

    Args:
        history: Anahtar olarak 'train_loss', 'val_loss',
                 'train_acc', 'val_acc' listelerini içeren sözlük.
        save_dir: Grafiklerin kaydedileceği dizin.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss Grafiği ---
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=4, label="Validation Loss")
    axes[0].set_title("Train ve Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy Grafiği ---
    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], "r-o", markersize=4, label="Validation Accuracy")
    axes[1].set_title("Train ve Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Eğitim grafiği kaydedildi: {path}")


def plot_confusion_matrix(cm: np.ndarray,
                           class_names: List[str],
                           save_dir: str = config.FIGURES_DIR) -> None:
    """
    Karışıklık matrisini ısı haritası olarak çizer ve kaydeder.

    Args:
        cm: Karışıklık matrisi (NxN numpy array).
        class_names: Sınıf isimleri listesi.
        save_dir: Görselin kaydedileceği dizin.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray", ax=ax
    )
    ax.set_title("Confusion Matrix (Karışıklık Matrisi)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Karışıklık matrisi kaydedildi: {path}")


def plot_class_distribution(data_counts: Dict[str, Dict[str, int]],
                              save_dir: str = config.FIGURES_DIR) -> None:
    """
    Veri seti sınıf dağılımını çubuk grafik olarak çizer.

    Args:
        data_counts: {'train': {'glioma': 100, ...}, 'val': {...}, 'test': {...}}
        save_dir: Görselin kaydedileceği dizin.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(config.CLASS_NAMES))
    width = 0.25

    splits = list(data_counts.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, split in enumerate(splits):
        counts = [data_counts[split].get(cls, 0) for cls in config.CLASS_NAMES]
        bars = ax.bar(x + i * width, counts, width, label=split.capitalize(), color=colors[i])
        # Çubukların üzerine sayı yaz
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontsize=9)

    ax.set_title("Sınıf Dağılımı (Train / Val / Test)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Görüntü Sayısı")
    ax.set_xticks(x + width)
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Sınıf dağılımı grafiği kaydedildi: {path}")


def plot_sample_images(dataset,
                        class_names: List[str],
                        num_samples: int = 4,
                        save_dir: str = config.FIGURES_DIR) -> None:
    """
    Her sınıftan örnek görüntüler gösterir ve kaydeder.

    Args:
        dataset: torchvision.datasets.ImageFolder nesnesi
                 (transform uygulanmamış veya denormalize edilmiş).
        class_names: Sınıf isimleri listesi.
        num_samples: Her sınıftan gösterilecek örnek sayısı.
        save_dir: Görselin kaydedileceği dizin.
    """
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, num_samples,
                              figsize=(3 * num_samples, 3 * num_classes))

    # Her sınıf için indeksleri topla
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    # ImageNet mean/std ile denormalizasyon fonksiyonu
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

    for class_idx in range(num_classes):
        indices = class_indices[class_idx][:num_samples]
        for j, sample_idx in enumerate(indices):
            img, _ = dataset[sample_idx]

            # Tensor ise denormalize et ve numpy'a çevir
            if isinstance(img, torch.Tensor):
                img = img * std + mean
                img = img.clamp(0, 1)
                img = img.permute(1, 2, 0).numpy()

            axes[class_idx, j].imshow(img)
            axes[class_idx, j].axis("off")
            if j == 0:
                axes[class_idx, j].set_title(
                    class_names[class_idx], fontsize=12, fontweight="bold"
                )

    plt.suptitle("Her Sınıftan Örnek Görüntüler", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_images.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Örnek görüntüler kaydedildi: {path}")


# =============================================================================
# Sonuç Kaydetme
# =============================================================================

def save_results_to_file(results: dict,
                          file_path: str) -> None:
    """
    Değerlendirme sonuçlarını metin dosyasına kaydeder.

    Args:
        results: Sonuç sözlüğü.
        file_path: Kaydedilecek dosya yolu.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Beyin Tumoru MRI Siniflandirma -- Degerlendirme Sonuclari\n")
        f.write("=" * 60 + "\n\n")

        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}:\n{value}\n\n")

    print(f"[SAVE] Sonuçlar kaydedildi: {file_path}")


# torch import — plot_sample_images içinde kullanılıyor
import torch
