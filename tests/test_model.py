"""tests/test_model.py — model.py testleri."""

import pytest
import torch

import config
from model import create_model, load_trained_model, get_model_summary


def test_create_model_output_shape():
    model = create_model(pretrained=False)
    assert model.fc.out_features == config.NUM_CLASSES


def test_create_model_forward_pass(tiny_model):
    x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    output = tiny_model(x)
    assert output.shape == (1, config.NUM_CLASSES)


def test_create_model_freeze_backbone():
    model = create_model(pretrained=False, freeze_backbone=True)
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in model.fc.parameters())


def test_load_trained_model_wrapped_checkpoint(tiny_checkpoint):
    model = load_trained_model(tiny_checkpoint, device=torch.device("cpu"))
    assert model.fc.out_features == config.NUM_CLASSES
    assert not model.training  # eval() modunda olmali


def test_load_trained_model_raw_state_dict(tmp_path, tiny_model):
    path = tmp_path / "raw_state_dict.pth"
    torch.save(tiny_model.state_dict(), path)
    model = load_trained_model(str(path), device=torch.device("cpu"))
    assert model.fc.out_features == config.NUM_CLASSES


def test_load_trained_model_missing_file():
    with pytest.raises(FileNotFoundError):
        load_trained_model("bu/yol/hic/olmayan_model.pth", device=torch.device("cpu"))


def test_get_model_summary(tiny_model):
    summary = get_model_summary(tiny_model)
    assert summary["total_params"] == summary["trainable_params"] + summary["frozen_params"]
    assert summary["total_params"] > 0
