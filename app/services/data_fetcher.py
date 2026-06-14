import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from app.config import settings

class DataFetcher:
    @staticmethod
    def get_historical_data() -> pd.DataFrame:
        ticker = yf.Ticker(settings.TICKER)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=800)
        df = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
        if df.empty:
            raise ValueError(f"No historical data returned for ticker {settings.TICKER}")
        df = df.dropna(subset=['Close'])
        return df

    @classmethod
    def prepare_context_and_history(cls):
        df = cls.get_historical_data()
        if len(df) < settings.CONTEXT_WINDOW:
            raise ValueError(f"Insufficient trading data. Expected at least {settings.CONTEXT_WINDOW} items, got {len(df)}")
        context_df = df.iloc[-settings.CONTEXT_WINDOW:]
        context_prices_raw = context_df['Close'].values
        context_prices_log = np.log(context_prices_raw)
        
        history_display_df = df.iloc[-60:]
        history_dates = history_display_df.index.strftime('%Y-%m-%d').tolist()
        history_prices = history_display_df['Close'].values.tolist()
        
        return {
            "context_prices_log": context_prices_log,
            "history_dates": history_dates,
            "history_prices": history_prices,
            "last_date": df.index[-1],
            "current_price": float(df['Close'].iloc[-1])
        }
