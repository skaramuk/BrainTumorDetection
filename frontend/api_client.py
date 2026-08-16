"""
frontend/api_client.py — FastAPI backend'ine HTTP uzerinden bagli ince istemci.

Streamlit tarafinda torch/model.py/predict.py/gradcam.py hicbir yerde
import edilmez; tum inference islemleri bu modul uzerinden backend'e
HTTP istegi olarak gonderilir.
"""

import os
from typing import Dict, List, Tuple

import requests

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

_TIMEOUT_SHORT = 5
_TIMEOUT_PREDICT = 60


class BackendError(Exception):
    """Backend'e ulasilamadiginda veya backend hata dondurdugunde firlatilir."""


def _handle_response(response: requests.Response) -> Dict:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise BackendError(f"[{response.status_code}] {detail}")
    return response.json()


def health() -> Dict:
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=_TIMEOUT_SHORT)
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Backend'e ulasilamiyor: {e}")
    return _handle_response(response)


def predict(file_bytes: bytes, filename: str) -> Dict:
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"image": (filename, file_bytes)},
            timeout=_TIMEOUT_PREDICT,
        )
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Backend'e ulasilamiyor: {e}")
    return _handle_response(response)


def predict_batch(files: List[Tuple[str, bytes]]) -> Dict:
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            files=[("images", (filename, file_bytes)) for filename, file_bytes in files],
            timeout=_TIMEOUT_PREDICT,
        )
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Backend'e ulasilamiyor: {e}")
    return _handle_response(response)


def gradcam(file_bytes: bytes, filename: str, predicted_class: str) -> Dict:
    try:
        response = requests.post(
            f"{BASE_URL}/gradcam",
            files={"image": (filename, file_bytes)},
            data={"predicted_class": predicted_class},
            timeout=_TIMEOUT_PREDICT,
        )
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Backend'e ulasilamiyor: {e}")
    return _handle_response(response)


def get_history() -> Dict:
    try:
        response = requests.get(f"{BASE_URL}/history", timeout=_TIMEOUT_SHORT)
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Backend'e ulasilamiyor: {e}")
    return _handle_response(response)


def clear_history() -> Dict:
    try:
        response = requests.delete(f"{BASE_URL}/history", timeout=_TIMEOUT_SHORT)
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Backend'e ulasilamiyor: {e}")
    return _handle_response(response)
