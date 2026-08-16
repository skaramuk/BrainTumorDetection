"""
model.py — Model tanımlama modülü.

Önceden eğitilmiş ResNet18 modelini yükler ve son fully connected
katmanını 4 sınıflı beyin tümörü sınıflandırmasına uyarlar.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import config


def create_model(num_classes: int = config.NUM_CLASSES,
                 pretrained: bool = True,
                 freeze_backbone: bool = False) -> nn.Module:
    """
    ResNet18 tabanlı transfer learning modeli oluşturur.

    Args:
        num_classes: Çıkış sınıf sayısı (varsayılan: 4).
        pretrained: ImageNet ağırlıklarını kullan (varsayılan: True).
        freeze_backbone: True ise yalnızca son katman eğitilir,
                         önceki katmanlar dondurulur (varsayılan: False).

    Returns:
        Uyarlanmış ResNet18 modeli.
    """
    # ImageNet üzerinde önceden eğitilmiş ağırlıkları yükle
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        print("[MODEL] ResNet18 -- ImageNet agirliklari yuklendi.")
    else:
        model = models.resnet18(weights=None)
        print("[MODEL] ResNet18 -- Rastgele agirliklarla baslatildi.")

    # İsteğe bağlı: Backbone katmanlarını dondur (feature extraction modu)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("[MODEL] Backbone katmanları donduruldu (feature extraction).")

    # Son fully connected katmanı sınıf sayısına göre değiştir
    # ResNet18'in son fc katmanı: Linear(512, 1000) -> Linear(512, num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(f"[MODEL] Son FC katmani guncellendi: {in_features} -> {num_classes}")

    return model


def load_trained_model(model_path: str,
                       num_classes: int = config.NUM_CLASSES,
                       device=config.DEVICE) -> nn.Module:
    """
    Kaydedilmiş model ağırlıklarını yükler.

    Args:
        model_path: Kaydedilmiş .pth dosyasının yolu.
        num_classes: Çıkış sınıf sayısı.
        device: Yükleme yapılacak cihaz (CPU/GPU).

    Returns:
        Ağırlıkları yüklenmiş model.

    Raises:
        FileNotFoundError: Model dosyası bulunamazsa.
    """
    import os
    import torch

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    # Boş model oluştur (pretrained=False çünkü kendi ağırlıklarımızı yükleyeceğiz)
    model = create_model(num_classes=num_classes, pretrained=False)

    # Kaydedilmiş checkpoint'i yükle
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Checkpoint formatını kontrol et (dict mi yoksa doğrudan state_dict mi)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", "?")
        print(f"[MODEL] Checkpoint yuklendi -- Epoch: {epoch}, Val Loss: {val_loss:.4f}"
              if isinstance(val_loss, float)
              else f"[MODEL] Checkpoint yuklendi -- Epoch: {epoch}")
    else:
        # Doğrudan state_dict kaydedilmişse
        model.load_state_dict(checkpoint)
        print("[MODEL] Model state_dict yüklendi.")

    model = model.to(device)
    model.eval()

    return model


def get_model_summary(model: nn.Module) -> dict:
    """
    Model hakkında özet bilgi döndürür.

    Args:
        model: PyTorch modeli.

    Returns:
        Toplam parametre sayısı ve eğitilebilir parametre sayısı.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }

    print(f"[MODEL] Toplam parametre  : {total_params:,}")
    print(f"[MODEL] Eğitilebilir      : {trainable_params:,}")
    print(f"[MODEL] Dondurulmuş       : {summary['frozen_params']:,}")

    return summary
