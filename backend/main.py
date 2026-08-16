"""
backend/main.py — NeuroScan AI FastAPI uygulaması.

Calistirma (repo kokunden):
    python -m uvicorn backend.main:app --reload --port 8000
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import config
from backend import db, inference
from backend.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    DeleteHistoryResponse,
    GradCamResponse,
    HealthResponse,
    HistoryItem,
    HistoryListResponse,
    PredictionResponse,
    ProbabilityItem,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    try:
        inference.load_model()
    except Exception as e:
        # Model yuklenemese bile API ayakta kalir; /health durumu False doner
        # ve /predict, /predict/batch, /gradcam 503 ile cevap verir.
        print(f"[STARTUP HATASI] Model yuklenemedi: {e}")
    yield


app = FastAPI(title="NeuroScan AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        raw = upload.file.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"Goruntu okunamadi: {upload.filename}"
        )


def _require_model_loaded() -> None:
    if not inference.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model yuklenmedi.")


def _to_prediction_response(filename: str, prediction: dict, row: dict) -> PredictionResponse:
    return PredictionResponse(
        id=row["id"],
        filename=filename,
        created_at=row["created_at"],
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        top2_class=prediction["top2_class"],
        top2_probability=prediction["top2_probability"],
        margin=prediction["margin"],
        status=prediction["status"],
        probabilities=[ProbabilityItem(**p) for p in prediction["probabilities"]],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=inference.is_model_loaded(),
        device=inference.get_device_str(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(image: UploadFile = File(...)) -> PredictionResponse:
    _require_model_loaded()
    pil_image = _read_image(image)

    prediction = inference.run_prediction(pil_image)
    row = db.insert_analysis(
        filename=image.filename,
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        top2_class=prediction["top2_class"],
        top2_probability=prediction["top2_probability"],
        margin=prediction["margin"],
        status=prediction["status"],
        probabilities=prediction["probabilities"],
    )
    return _to_prediction_response(image.filename, prediction, row)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(images: List[UploadFile] = File(...)) -> BatchPredictionResponse:
    _require_model_loaded()

    results: List[BatchPredictionItem] = []
    for upload in images:
        try:
            pil_image = _read_image(upload)
        except HTTPException as e:
            results.append(
                BatchPredictionItem(filename=upload.filename, success=False, error=e.detail)
            )
            continue

        try:
            prediction = inference.run_prediction(pil_image)
            row = db.insert_analysis(
                filename=upload.filename,
                predicted_class=prediction["predicted_class"],
                confidence=prediction["confidence"],
                top2_class=prediction["top2_class"],
                top2_probability=prediction["top2_probability"],
                margin=prediction["margin"],
                status=prediction["status"],
                probabilities=prediction["probabilities"],
            )
            results.append(
                BatchPredictionItem(
                    filename=upload.filename,
                    success=True,
                    result=_to_prediction_response(upload.filename, prediction, row),
                )
            )
        except Exception as e:
            results.append(
                BatchPredictionItem(filename=upload.filename, success=False, error=str(e))
            )

    return BatchPredictionResponse(results=results)


@app.post("/gradcam", response_model=GradCamResponse)
def gradcam(
    image: UploadFile = File(...),
    predicted_class: str = Form(...),
) -> GradCamResponse:
    _require_model_loaded()
    pil_image = _read_image(image)

    try:
        heatmap_b64, overlay_b64, class_idx = inference.run_gradcam(pil_image, predicted_class)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return GradCamResponse(
        predicted_class=predicted_class,
        class_index=class_idx,
        heatmap_png_base64=heatmap_b64,
        overlay_png_base64=overlay_b64,
    )


@app.get("/history", response_model=HistoryListResponse)
def get_history() -> HistoryListResponse:
    rows = db.get_history()
    items = [
        HistoryItem(
            id=row["id"],
            created_at=row["created_at"],
            filename=row["filename"],
            predicted_class=row["predicted_class"],
            confidence=row["confidence"],
            top2_class=row["top2_class"],
            margin=row["margin"],
            status=row["status"],
        )
        for row in rows
    ]
    return HistoryListResponse(items=items, total=len(items))


@app.delete("/history", response_model=DeleteHistoryResponse)
def clear_history() -> DeleteHistoryResponse:
    deleted = db.clear_history()
    return DeleteHistoryResponse(deleted=deleted)
