"""
train.py — Eğitim ve doğrulama döngüsü modülü.

Eğitim döngüsü, early stopping, learning rate scheduler ve
en iyi modeli kaydetme işlemlerini yürütür.
"""

import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Validation loss iyileşmezse eğitimi erken durdurur.

    Attributes:
        patience: İyileşme olmadan beklenecek epoch sayısı.
        min_delta: Minimum iyileşme eşiği.
        counter: Son iyileşmeden bu yana geçen epoch sayısı.
        best_loss: Şimdiye kadarki en iyi validation loss.
        should_stop: Eğitimin durması gerekip gerekmediği.
    """

    def __init__(self,
                 patience: int = config.EARLY_STOPPING_PATIENCE,
                 min_delta: float = config.EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Yeni validation loss ile güncelleme yapar.

        Args:
            val_loss: Mevcut epoch'un validation loss değeri.

        Returns:
            True ise eğitim durdurulmalı.
        """
        if val_loss < self.best_loss - self.min_delta:
            # İyileşme var → sayacı sıfırla
            self.best_loss = val_loss
            self.counter = 0
        else:
            # İyileşme yok → sayacı artır
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[EARLY STOP] {self.patience} epoch boyunca iyileşme olmadı. "
                      f"Eğitim durduruluyor.")

        return self.should_stop


# =============================================================================
# Tek Epoch Eğitimi
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Bir epoch boyunca modeli eğitir.

    Args:
        model: Eğitilecek model.
        dataloader: Eğitim DataLoader'ı.
        criterion: Loss fonksiyonu.
        optimizer: Optimizer.
        device: Hesaplama cihazı.

    Returns:
        (ortalama_loss, accuracy_yüzdesi) tuple'ı.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="  Train", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrikleri güncelle
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress bar güncelle
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Model performansını validation seti üzerinde değerlendirir.

    Args:
        model: Değerlendirilecek model.
        dataloader: Validation DataLoader'ı.
        criterion: Loss fonksiyonu.
        device: Hesaplama cihazı.

    Returns:
        (ortalama_loss, accuracy_yüzdesi) tuple'ı.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="  Val  ", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# =============================================================================
# Tam Eğitim Döngüsü
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = config.DEVICE,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    save_path: str = config.BEST_MODEL_PATH,
    class_names: list = None,
) -> Dict[str, List[float]]:
    """
    Modeli tam eğitim döngüsüyle eğitir.

    Özellikler:
        - AdamW optimizer
        - ReduceLROnPlateau scheduler
        - Early stopping
        - En iyi modeli kaydetme

    Args:
        model: Eğitilecek model.
        train_loader: Eğitim DataLoader'ı.
        val_loader: Validation DataLoader'ı.
        device: Hesaplama cihazı.
        epochs: Maksimum epoch sayısı.
        learning_rate: Başlangıç öğrenme oranı.
        weight_decay: Ağırlık azaltma katsayısı.
        save_path: En iyi modelin kaydedileceği yol.
        class_names: Sınıf isimleri (checkpoint'e kaydedilir).

    Returns:
        Eğitim geçmişi sözlüğü:
        {'train_loss': [...], 'val_loss': [...],
         'train_acc': [...], 'val_acc': [...]}
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    # Model'i cihaza taşı
    model = model.to(device)

    # Loss, optimizer ve scheduler tanımla
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    # Early stopping
    early_stopping = EarlyStopping()

    # Eğitim geçmişi
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0

    print("\n" + "=" * 70)
    print(f"  Egitim Basliyor -- Cihaz: {device}")
    print(f"  Epochs: {epochs} | LR: {learning_rate} | Batch: {train_loader.batch_size}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Eğitim
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler güncelle
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # Geçmişe ekle
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        # Epoch sonuçlarını yazdır
        lr_info = f" (LR: {old_lr:.2e}->{new_lr:.2e})" if new_lr != old_lr else ""
        improved = " *" if val_loss < best_val_loss else ""

        print(f"Epoch [{epoch:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Süre: {epoch_time:.1f}s{lr_info}{improved}")

        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Checkpoint kaydet
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "class_names": class_names,
            }
            torch.save(checkpoint, save_path)

        # Early stopping kontrolü
        if early_stopping(val_loss):
            break

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"  Egitim Tamamlandi -- Toplam Sure: {total_time:.1f}s")
    print(f"  En iyi epoch: {best_epoch} | Val Loss: {best_val_loss:.4f}")
    print(f"  Model kaydedildi: {save_path}")
    print("=" * 70)

    return history
