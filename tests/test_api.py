"""tests/test_api.py — backend/main.py FastAPI endpoint testleri (TestClient ile)."""

import io

from PIL import Image

import config


def _image_bytes(color=(100, 120, 140), fmt="JPEG"):
    buf = io.BytesIO()
    Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), color=color).save(buf, format=fmt)
    return buf.getvalue()


def test_health_without_model(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False


def test_health_with_model(api_client_with_model):
    response = api_client_with_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_503_without_model(api_client):
    response = api_client.post("/predict", files={"image": ("test.jpg", _image_bytes(), "image/jpeg")})
    assert response.status_code == 503


def test_predict_success(api_client_with_model):
    response = api_client_with_model.post(
        "/predict", files={"image": ("test.jpg", _image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in config.CLASS_NAMES
    assert len(body["probabilities"]) == config.NUM_CLASSES
    assert body["id"] == 1

    history_response = api_client_with_model.get("/history")
    assert history_response.json()["total"] == 1


def test_predict_bad_image_returns_400(api_client_with_model):
    response = api_client_with_model.post(
        "/predict", files={"image": ("bad.jpg", b"gecerli bir goruntu degil", "image/jpeg")}
    )
    assert response.status_code == 400


def test_predict_batch_mixed_success_and_failure(api_client_with_model):
    response = api_client_with_model.post(
        "/predict/batch",
        files=[
            ("images", ("good.jpg", _image_bytes(), "image/jpeg")),
            ("images", ("bad.jpg", b"bozuk veri", "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    by_name = {r["filename"]: r for r in results}
    assert by_name["good.jpg"]["success"] is True
    assert by_name["bad.jpg"]["success"] is False


def test_gradcam_success(api_client_with_model):
    response = api_client_with_model.post(
        "/gradcam",
        files={"image": ("test.jpg", _image_bytes(), "image/jpeg")},
        data={"predicted_class": config.CLASS_NAMES[0]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["heatmap_png_base64"]) > 0
    assert len(body["overlay_png_base64"]) > 0


def test_gradcam_unknown_class_returns_400(api_client_with_model):
    response = api_client_with_model.post(
        "/gradcam",
        files={"image": ("test.jpg", _image_bytes(), "image/jpeg")},
        data={"predicted_class": "gecersiz_sinif"},
    )
    assert response.status_code == 400


def test_history_delete(api_client_with_model):
    api_client_with_model.post("/predict", files={"image": ("a.jpg", _image_bytes(), "image/jpeg")})
    api_client_with_model.post("/predict", files={"image": ("b.jpg", _image_bytes(), "image/jpeg")})

    delete_response = api_client_with_model.delete("/history")
    assert delete_response.json()["deleted"] == 2

    history_response = api_client_with_model.get("/history")
    assert history_response.json()["total"] == 0
