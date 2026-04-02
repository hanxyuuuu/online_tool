from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .inference import predict_sequences
from .model_loader import MODEL_SCRIPT_PATH, MODEL_WEIGHTS_PATH, get_loaded_model
from .schemas import PredictionRequest, PredictionResponse


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="TF-DNA Inference Demo")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def load_model_on_startup() -> None:
    get_loaded_model()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "dna_expected_length": 101,
            "protein_max_length": 800,
            "model_script_path": str(MODEL_SCRIPT_PATH),
            "model_weights_path": str(MODEL_WEIGHTS_PATH),
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    get_loaded_model()
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    return predict_sequences(payload.dna_sequence, payload.protein_sequence)
