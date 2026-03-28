import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import FloodPredictionInput, FloodPredictionOutput

# Global variable to hold the loaded XGBoost model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        print("Loading XGBoost model...")
        model = joblib.load("xgboost_flood_model.pkl")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    # Clean up (if needed)
    model = None

app = FastAPI(
    title="Flood Prediction API",
    description="API for predicting incoming normalized water level using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict", response_model=FloodPredictionOutput)
async def predict_water_level(data: FloodPredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Create DataFrame containing EXACTLY the features the model was trained on
        # Features from train_xgboost.py: ['Mực nước (m)', 'Month', 'Rolling_Mean_7d', 'Delta_1d', 'Dung tích (m3)', 'Q đến (m3/s)', 'Q xả (m3/s)']
        input_data = pd.DataFrame([{
            'Mực nước (m)': data.muc_nuoc,
            'Month': data.month,
            'Rolling_Mean_7d': data.rolling_mean_7d,
            'Delta_1d': data.delta_1d,
            'Dung tích (m3)': data.dung_tich,
            'Q đến (m3/s)': data.q_den,
            'Q xả (m3/s)': data.q_xa
        }])
        
        prediction = model.predict(input_data)
        
        return FloodPredictionOutput(
            predicted_muc_nuoc_t_plus_1=float(prediction[0]),
            lat=data.lat,
            long=data.long
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}
