"""
Blood Cell Anomaly Detection - FastAPI Web Application
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import load_model, predict, DISEASE_CATEGORIES
from app.preprocessing import (
    fit_and_save_scaler, load_scaler, preprocess_input,
    CELL_TYPES, AGE_MAPPING, SEX_MAPPING, FEATURE_NAMES
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Blood_Cell_Anomaly_Detection.pth"
CSV_PATH = BASE_DIR / "blood_cell_anomaly_detection.csv"

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Blood Cell Anomaly Detection",
    description="Kan hücresi anomali tespiti için yapay zeka destekli web uygulaması",
    version="1.0.0"
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# ── Startup: load model & scaler ──────────────────────────────────────────
model = None
scaler = None


@app.on_event("startup")
async def startup_event():
    global model, scaler

    # Load PyTorch model
    print("🔬 Model yükleniyor...")
    model = load_model(str(MODEL_PATH))
    print("✅ Model başarıyla yüklendi!")

    # Load or fit scaler
    scaler = load_scaler()
    if scaler is None:
        print("📊 Scaler eğitim verisinden oluşturuluyor...")
        scaler = fit_and_save_scaler(str(CSV_PATH))
        print("✅ Scaler başarıyla oluşturuldu ve kaydedildi!")
    else:
        print("✅ Scaler başarıyla yüklendi!")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with the prediction form."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cell_types": CELL_TYPES,
        "age_groups": list(AGE_MAPPING.keys()),
        "sex_options": list(SEX_MAPPING.keys()),
    })


@app.post("/predict")
async def predict_endpoint(request: Request):
    """Process form data and return prediction."""
    try:
        form_data = await request.form()
        data = dict(form_data)

        # Preprocess input
        scaled_features = preprocess_input(data, scaler)

        # Run inference
        result = predict(model, scaled_features)

        return JSONResponse(content={
            "success": True,
            "result": result
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None, "scaler_loaded": scaler is not None}
