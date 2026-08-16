"""
backend/inference.py — Model ve Grad-CAM singleton'ları, tahmin/ısı haritası
orkestrasyonu ve güven durumu (confidence status) iş mantığı.

Model ve GradCAM nesnesi süreç ömrü boyunca yalnızca BİR KEZ oluşturulur
(bkz. load_model). Bu, eski Streamlit uygulamasındaki her tıklamada yeni bir
GradCAM örneği (ve dolayısıyla yeni forward/backward hook'lar) oluşturup
eskilerini hiç temizlememe hatasını kökten ortadan kaldırır.
"""

import base64
import io
import threading
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

import config
from gradcam import GradCAM, apply_colormap_on_image
from model import load_trained_model
from predict import predict_image

_model = None
_device = None
_gradcam: Optional[GradCAM] = None
_gradcam_lock = threading.Lock()

_eval_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])


def load_model() -> None:
    """Modeli ve Grad-CAM örneğini süreç başlangıcında bir kez yükler."""
    global _model, _device, _gradcam

    _device = config.DEVICE
    _model = load_trained_model(config.BEST_MODEL_PATH, device=_device)

    target_layer = _model.layer4[-1]
    _gradcam = GradCAM(_model, target_layer)
    print("[INFERENCE] Model ve Grad-CAM hazir.")


def is_model_loaded() -> bool:
    return _model is not None


def get_device_str() -> str:
    return str(_device) if _device is not None else str(config.DEVICE)


def determine_status(confidence: float, margin: float) -> str:
    """Tahmin güvenilirliğini eşik değerlerine göre sınıflandırır."""
    if confidence < config.CONFIDENCE_THRESHOLD or margin < config.MARGIN_THRESHOLD:
        return "Review Required"
    return "High Confidence"


def run_prediction(image: Image.Image) -> Dict:
    """
    Bir görüntüyü sınıflandırır ve durum/margin bilgileriyle zenginleştirilmiş
    bir sonuç sözlüğü döndürür.
    """
    with torch.inference_mode():
        result = predict_image(image=image, model=_model, device=_device)

    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    top1_class, top1_prob = sorted_probs[0]
    top2_class, top2_prob = sorted_probs[1]

    confidence = float(result["confidence"])
    top2_prob = float(top2_prob)
    margin = confidence - top2_prob

    return {
        "predicted_class": result["predicted_class"],
        "confidence": confidence,
        "top2_class": top2_class,
        "top2_probability": top2_prob,
        "margin": margin,
        "status": determine_status(confidence, margin),
        "probabilities": [
            {"class_name": k, "probability": float(v)} for k, v in sorted_probs
        ],
    }


def _pil_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_gradcam(image: Image.Image, predicted_class: str) -> Tuple[str, str, int]:
    """
    Verilen görüntü için Grad-CAM ısı haritası ve overlay görüntüsünü üretir.

    Returns:
        (heatmap_png_base64, overlay_png_base64, class_index)
    """
    if predicted_class not in config.CLASS_NAMES:
        raise ValueError(f"Bilinmeyen sinif adi: {predicted_class}")

    class_idx = config.CLASS_NAMES.index(predicted_class)
    input_tensor = _eval_transform(image).unsqueeze(0).to(_device)

    with _gradcam_lock:
        cam = _gradcam.generate_cam(input_tensor, target_class=class_idx)
        heatmap_img, overlay_img = apply_colormap_on_image(
            image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), cam
        )

    return _pil_to_b64_png(heatmap_img), _pil_to_b64_png(overlay_img), class_idx
