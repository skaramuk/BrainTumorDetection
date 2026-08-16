"""
backend/schemas.py — API istek/yanıt modelleri (Pydantic).
"""

from typing import List, Optional

from pydantic import BaseModel


class ProbabilityItem(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    top2_class: str
    top2_probability: float
    margin: float
    status: str
    probabilities: List[ProbabilityItem]
    created_at: str


class BatchPredictionItem(BaseModel):
    filename: str
    success: bool
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    results: List[BatchPredictionItem]


class GradCamResponse(BaseModel):
    predicted_class: str
    class_index: int
    heatmap_png_base64: str
    overlay_png_base64: str


class HistoryItem(BaseModel):
    id: int
    created_at: str
    filename: str
    predicted_class: str
    confidence: float
    top2_class: str
    margin: float
    status: str


class HistoryListResponse(BaseModel):
    items: List[HistoryItem]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class DeleteHistoryResponse(BaseModel):
    deleted: int
