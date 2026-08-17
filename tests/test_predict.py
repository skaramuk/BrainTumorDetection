"""tests/test_predict.py — predict.py testleri."""

import pytest
import torch

import config
from predict import predict_image, predict_single_image


def test_predict_image_output_shape(tiny_model, sample_image):
    result = predict_image(image=sample_image, model=tiny_model, device=torch.device("cpu"))

    assert set(result.keys()) == {"predicted_class", "predicted_index", "confidence", "probabilities"}
    assert result["predicted_class"] in config.CLASS_NAMES
    assert 0 <= result["predicted_index"] < config.NUM_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == set(config.CLASS_NAMES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_image_without_model_raises_if_no_checkpoint(sample_image):
    with pytest.raises(FileNotFoundError):
        predict_image(image=sample_image, model=None, model_path="hic/olmayan/model.pth")


def test_predict_single_image_end_to_end(tmp_path, tiny_checkpoint, sample_image):
    image_path = tmp_path / "test_mri.jpg"
    sample_image.save(image_path)

    result = predict_single_image(str(image_path), model_path=tiny_checkpoint, device=torch.device("cpu"))

    assert result["predicted_class"] in config.CLASS_NAMES
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_single_image_missing_file(tiny_checkpoint):
    with pytest.raises(FileNotFoundError):
        predict_single_image("hic/olmayan/goruntu.jpg", model_path=tiny_checkpoint)
