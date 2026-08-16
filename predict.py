"""
predict.py — Tek görüntü tahmini modülü.

Kullanıcının verdiği bir MRI görüntüsünü sınıflandırır.
Tahmin edilen sınıfı ve tüm sınıfların olasılıklarını ekrana yazdırır.

Kullanım:
    python predict.py --image path/to/mri_image.jpg
    python predict.py --image path/to/mri_image.jpg --model models/best_model.pth
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

import config
from model import load_trained_model
from dataset import get_eval_transform


def predict_image(
    image: Image.Image,
    model: torch.nn.Module = None,
    model_path: str = config.BEST_MODEL_PATH,
    device: torch.device = config.DEVICE,
) -> dict:
    """
    Bir PIL Image nesnesini sınıflandırır.
    
    Args:
        image: Sınıflandırılacak PIL Image nesnesi.
        model: Yüklü model nesnesi (varsa). Yoksa model_path'ten yüklenir.
        model_path: Kaydedilmiş model dosyasının yolu (model verilmemişse kullanılır).
        device: Hesaplama cihazı.

    Returns:
        Tahmin sonuçlarını içeren sözlük:
        {
            'predicted_class': str,
            'predicted_index': int,
            'confidence': float,
            'probabilities': {class_name: probability, ...}
        }
    """
    if model is None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        model = load_trained_model(model_path, device=device)

    # Sınıf isimlerini al
    class_names = config.CLASS_NAMES
    try:
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "class_names" in checkpoint:
                class_names = checkpoint["class_names"]
    except:
        pass

    transform = get_eval_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze()

    # Olasılıklar tek boyutlu olmalı
    if probabilities.dim() == 0:
        probabilities = probabilities.unsqueeze(0)

    predicted_index = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item()

    prob_dict = {
        class_names[i]: probabilities[i].item()
        for i in range(len(class_names))
    }

    result = {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities": prob_dict,
    }

    return result


def predict_single_image(
    image_path: str,
    model_path: str = config.BEST_MODEL_PATH,
    device: torch.device = config.DEVICE,
) -> dict:
    """
    Tek bir MRI görüntüsünü sınıflandırır.

    Args:
        image_path: Görüntü dosyasının yolu.
        model_path: Kaydedilmiş model dosyasının yolu.
        device: Hesaplama cihazı.

    Returns:
        Tahmin sonuçlarını içeren sözlük.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Goruntu okunamadi: {image_path} - {e}")

    return predict_image(image, model=None, model_path=model_path, device=device)


def print_prediction(result: dict) -> None:
    """
    Tahmin sonuçlarını düzenli biçimde ekrana yazdırır.

    Args:
        result: predict_single_image çıktısı.
    """
    print("\n" + "=" * 50)
    print("  MRI GÖRÜNTÜ TAHMİN SONUCU")
    print("=" * 50)
    print(f"\n  Tahmin Edilen Sınıf : {result['predicted_class']}")
    print(f"  Güven Skoru         : {result['confidence']:.4f} "
          f"({result['confidence'] * 100:.2f}%)")
    print("\n  Sınıf Olasılıkları:")
    print("  " + "-" * 35)

    # Olasılıkları büyükten küçüğe sırala
    sorted_probs = sorted(
        result["probabilities"].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for cls_name, prob in sorted_probs:
        bar = "#" * int(prob * 30)  # Basit cubuk grafik
        marker = " <" if cls_name == result["predicted_class"] else ""
        print(f"  {cls_name:15s}: {prob:.4f} ({prob * 100:5.2f}%) {bar}{marker}")

    print("=" * 50)

    # Tıbbi uyarı
    print("\n  UYARI: Bu tahmin yalnizca akademik amaclidir.")
    print("  Bu sonuç tıbbi tanı yerine kullanılamaz.")
    print("  Tıbbi değerlendirme için uzman hekime başvurun.\n")


# =============================================================================
# Komut Satırı Arayüzü
# =============================================================================

def main():
    """Komut satırından tek görüntü tahmini yapar."""
    parser = argparse.ArgumentParser(
        description="Beyin Tümörü MRI Görüntü Sınıflandırma — Tek Görüntü Tahmini"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Tahmin edilecek MRI görüntüsünün dosya yolu.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=config.BEST_MODEL_PATH,
        help=f"Kullanılacak model dosyası (varsayılan: {config.BEST_MODEL_PATH}).",
    )

    args = parser.parse_args()

    try:
        result = predict_single_image(args.image, args.model)
        print_prediction(result)
    except FileNotFoundError as e:
        print(f"\n[HATA] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[HATA] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[HATA] Beklenmeyen hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
