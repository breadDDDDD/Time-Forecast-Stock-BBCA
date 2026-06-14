import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from chronos import ChronosBoltPipeline
from app.config import settings

class InferenceEngine:
    def __init__(self):
        self.pipeline = None

    def load_model(self):
        print(f"Initializing Inference Core: Pulling {settings.HF_MODEL_ID} from Hugging Face...")
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            settings.HF_MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        print("Chronos Engine successfully initialized into application lifecycle context.")

    def predict_horizon(self, context_prices_log: np.ndarray, last_date: datetime) -> list:
        if self.pipeline is None:
            raise RuntimeError("Inference system offline: Chronos pipeline has not been booted.")
        context_tensor = torch.tensor(context_prices_log, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            forecast = self.pipeline.predict(context_tensor, prediction_length=settings.HORIZON)
        forecast_samples = forecast[0].numpy()
        median_forecast_log = np.median(forecast_samples, axis=0)
        reconstructed_prices = np.exp(median_forecast_log)
        
        forecast_points = []
        current_date = last_date
        for i in range(settings.HORIZON):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            forecast_points.append({
                "day_index": i + 1,
                "date": current_date.strftime('%Y-%m-%d'),
                "value": round(float(reconstructed_prices[i]), 2)
            })
        return forecast_points

inference_engine = InferenceEngine()
