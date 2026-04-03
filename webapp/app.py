from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .emailer import send_prediction_email, validate_email_address
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
        request,
        "index.html",
        {
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
    recipient = None
    if payload.email is not None and payload.email.strip():
        try:
            recipient = validate_email_address(payload.email)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = predict_sequences(payload.dna_sequence, payload.protein_sequence)
    result.email_requested = recipient is not None

    if recipient is None:
        result.email_delivery_status = "not_requested"
        result.email_delivery_message = "No email address was provided. Result delivery was skipped."
        return result

    try:
        status, message = send_prediction_email(recipient, result)
        result.email_delivery_status = status
        result.email_delivery_message = message
    except Exception as exc:
        result.email_delivery_status = "failed"
        result.email_delivery_message = f"Prediction finished, but email delivery failed: {exc}"

    return result
