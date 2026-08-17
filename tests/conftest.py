"""
tests/conftest.py — paylaşılan pytest fixture'ları.

Hiçbir test gerçek egitilmis model dosyasina (models/best_model.pth) ya da
gercek veri setine (data/) ihtiyac duymaz -- ikisi de .gitignore'da, CI'de
mevcut degil. Onun yerine kucuk, egitilmemis modeller ve sentetik goruntuler
kullanilir.
"""

import torch
import pytest
from PIL import Image
from fastapi.testclient import TestClient

import config
from model import create_model
from gradcam import GradCAM
from backend import db, inference


@pytest.fixture(scope="session")
def tiny_model():
    """pretrained=False -> internet gerektirmez, rastgele agirlikli hizli model."""
    return create_model(pretrained=False)


@pytest.fixture
def sample_image():
    return Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), color=(100, 120, 140))


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_model):
    """load_trained_model'in diskten yukleme yolunu test etmek icin gercek bir gecici .pth dosyasi."""
    path = tmp_path / "fake_model.pth"
    torch.save({
        "model_state_dict": tiny_model.state_dict(),
        "class_names": config.CLASS_NAMES,
        "epoch": 1,
        "val_loss": 0.1234,
    }, path)
    return str(path)


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    """ImageFolder uyumlu kucuk bir agac: train/val/test altinda config.CLASS_NAMES'teki
    4 gercek sinif, her birinde 2 kucuk uretilmis goruntu.

    Dosya adlarinin icine split+sinif gomulu (orn. "train_glioma_0.jpg") --
    verify_no_data_leakage yalnizca dosya ADINA (tam yola degil) bakiyor, bu
    yuzden farkli split'lerdeki dosyalar yanlislikla ayni basename'e sahip
    olursa "temiz" senaryoda bile sahte bir sizinti tespit edilir.
    """
    for split in ("train", "val", "test"):
        for cls in config.CLASS_NAMES:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (32, 32), color=(i * 40, 60, 90)).save(d / f"{split}_{cls}_{i}.jpg")
    return tmp_path


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "history_test.db"))
    db.init_db()
    return config.DB_PATH


@pytest.fixture
def inference_with_tiny_model(monkeypatch, tiny_model):
    monkeypatch.setattr(inference, "_model", tiny_model)
    monkeypatch.setattr(inference, "_device", torch.device("cpu"))
    monkeypatch.setattr(inference, "_gradcam", GradCAM(tiny_model, tiny_model.layer4[-1]))
    return tiny_model


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "api_test.db"))
    # Gelistirme makinesinde models/best_model.pth gercek ve 44MB -- bu olmadan
    # her api_client testi gercek modeli yuklemeye calisirdi. Var olmayan bir
    # yola cekerek her iki ortamda da (yerel + CI) load_model()'in
    # FileNotFoundError ile deterministik sekilde basarisiz olmasini sagliyoruz
    # (backend/main.py'nin lifespan'i bunu zaten yakalayip _model=None birakiyor).
    monkeypatch.setattr(config, "BEST_MODEL_PATH", str(tmp_path / "no_such_model.pth"))
    from backend.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_client_with_model(api_client, tiny_model, monkeypatch):
    monkeypatch.setattr(inference, "_model", tiny_model)
    monkeypatch.setattr(inference, "_device", torch.device("cpu"))
    monkeypatch.setattr(inference, "_gradcam", GradCAM(tiny_model, tiny_model.layer4[-1]))
    return api_client
