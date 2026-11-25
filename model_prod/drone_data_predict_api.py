from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL_PATH = Path(__file__).resolve().parent / "model_outputs" / "best_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Drone Battery Temperature API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    model = app.state.model

    try:
        frame = pd.read_csv(BytesIO(file.file.read()))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail="Upload a valid CSV.") from exc

    if frame.empty:
        raise HTTPException(status_code=422, detail="The CSV has no data rows.")

    # Use the same normalization as transform_training_data().
    frame.columns = (
        frame.columns.str.lower()
        .str.strip()
        .str.replace("(", " ", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(")", " ", regex=False)
        .str.strip()
    )

    required = list(model.feature_names_in_)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing},
        )

    features = frame[required].copy()
    try:
        predictions = model.predict(features)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="CSV feature values do not match the model input types.",
        ) from exc

    return {"predicted_temperature_f": predictions.tolist()}