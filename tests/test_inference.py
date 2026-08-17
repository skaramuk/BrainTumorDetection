"""tests/test_inference.py — backend/inference.py testleri (saf mantik + run_prediction)."""

import pytest
import torch

import config
from backend import inference


@pytest.mark.parametrize("confidence,margin,expected", [
    (0.95, 0.50, "High Confidence"),
    (0.79, 0.50, "Review Required"),   # confidence esigin hemen altinda
    (0.95, 0.19, "Review Required"),   # margin esigin hemen altinda
    (0.50, 0.05, "Review Required"),   # ikisi de kotu
])
def test_determine_status(confidence, margin, expected):
    assert inference.determine_status(confidence, margin) == expected


def test_determine_status_at_threshold_boundary():
    # kod strict `<` kullaniyor, yani esige TAM esit olmak "kotu" sayilmaz
    assert inference.determine_status(config.CONFIDENCE_THRESHOLD, 0.50) == "High Confidence"
    assert inference.determine_status(0.95, config.MARGIN_THRESHOLD) == "High Confidence"


def test_is_model_loaded_reflects_module_state(inference_with_tiny_model):
    assert inference.is_model_loaded() is True


def test_run_prediction_shape(inference_with_tiny_model, sample_image):
    result = inference.run_prediction(sample_image)

    assert result["predicted_class"] in config.CLASS_NAMES
    assert result["status"] in ("High Confidence", "Review Required")
    assert len(result["probabilities"]) == config.NUM_CLASSES
    total = sum(p["probability"] for p in result["probabilities"])
    assert total == pytest.approx(1.0, abs=1e-4)


def test_run_gradcam_unknown_class_raises(inference_with_tiny_model, sample_image):
    with pytest.raises(ValueError):
        inference.run_gradcam(sample_image, "bilinmeyen_sinif")


def test_run_gradcam_returns_base64_pngs(inference_with_tiny_model, sample_image):
    heatmap_b64, overlay_b64, class_idx = inference.run_gradcam(sample_image, config.CLASS_NAMES[0])

    assert isinstance(heatmap_b64, str) and len(heatmap_b64) > 0
    assert isinstance(overlay_b64, str) and len(overlay_b64) > 0
    assert class_idx == 0
