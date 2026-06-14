from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.data_fetcher import DataFetcher
from app.services.model_inference import inference_engine

router = APIRouter(prefix="/api/v1", tags=["Inference Matrix"])

class ForecastItem(BaseModel):
    day_index: int
    date: str
    value: float

class ForecastResponse(BaseModel):
    ticker: str
    current_price: float
    history_dates: List[str]
    history_prices: List[float]
    forecast: List[ForecastItem]

@router.get("/forecast", response_model=ForecastResponse)
async def get_chronos_forecast():
    try:
        market_data = DataFetcher.prepare_context_and_history()
        predictions = inference_engine.predict_horizon(
            context_prices_log=market_data["context_prices_log"],
            last_date=market_data["last_date"]
        )
        return ForecastResponse(
            ticker="BBCA.JK",
            current_price=market_data["current_price"],
            history_dates=market_data["history_dates"],
            history_prices=market_data["history_prices"],
            forecast=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failure: {str(e)}")
