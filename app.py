"""
BBCA Stock Forecast — fine-tuned Chronos model demo.

Pulls live BBCA.JK history from Yahoo Finance, feeds the last CONTEXT
closing prices (log-transformed, matching how the model was trained)
into the fine-tuned Chronos model, and shows the HORIZON-day forecast.
"""

import datetime as dt
import os

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yfinance as yf
from chronos import BaseChronosPipeline

MODEL_ID = "SkibidiBreaddd/BBCA-Chronos-14062026-v0"
TICKER = "BBCA.JK"
CONTEXT = 512
HORIZON = 5

HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if the model repo is private/gated

_pipeline = None


def get_pipeline():
    """Load the Chronos pipeline once and reuse it across requests."""
    global _pipeline
    if _pipeline is None:
        kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        _pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.float32,
            **kwargs,
        )
    return _pipeline


def fetch_history(min_points=CONTEXT + 30):
    """Pull enough daily bars from Yahoo Finance to cover the model's context window."""
    data = yf.download(TICKER, period="5y", interval="1d", progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty or len(data) < min_points:
        data = yf.download(TICKER, period="max", interval="1d", progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    return data.dropna(subset=["Close"])


def next_business_days(start_date, n):
    """Approximate the next n trading days (weekdays only — IDX holidays aren't modeled)."""
    days, d = [], start_date
    while len(days) < n:
        d = d + dt.timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


def run_forecast():
    history = fetch_history()
    if len(history) < CONTEXT:
        raise gr.Error(
            f"Only found {len(history)} trading days of {TICKER} history, "
            f"need at least {CONTEXT}."
        )

    close = history["Close"].values.astype(np.float64)
    log_close = np.log(close)
    context_window = torch.tensor(log_close[-CONTEXT:], dtype=torch.float32)

    pipeline = get_pipeline()
    quantiles, _mean = pipeline.predict_quantiles(
        context=context_window,
        prediction_length=HORIZON,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    # quantiles shape: [1, HORIZON, 3] -> (q10, q50, q90), still in log-price space
    q10 = np.exp(quantiles[0, :, 0].numpy())
    median = np.exp(quantiles[0, :, 1].numpy())
    q90 = np.exp(quantiles[0, :, 2].numpy())

    last_close = float(close[-1])
    last_date = history.index[-1].to_pydatetime()
    forecast_dates = next_business_days(last_date, HORIZON)

    # ---- table ----
    rows, prev = [], last_close
    for d, lo, mid, hi in zip(forecast_dates, q10, median, q90):
        pct = (mid - prev) / prev * 100
        rows.append(
            {
                "Date": d.strftime("%Y-%m-%d"),
                "Predicted Close (IDR)": round(float(mid), 2),
                "Low (10%)": round(float(lo), 2),
                "High (90%)": round(float(hi), 2),
                "Change vs prev day": f"{pct:+.2f}%",
            }
        )
        prev = mid
    table = pd.DataFrame(rows)

    # ---- chart ----
    hist_window = history.tail(60)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_window.index,
            y=hist_window["Close"],
            mode="lines",
            name="Historical close",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_date] + forecast_dates,
            y=[last_close] + list(median),
            mode="lines+markers",
            name="Forecast (median)",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=list(q90) + list(q10[::-1]),
            fill="toself",
            fillcolor="rgba(214,39,40,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% interval",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"{TICKER} — last 60 trading days + {HORIZON}-day forecast",
        xaxis_title="Date",
        yaxis_title="Price (IDR)",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=40),
    )

    final_price = median[-1]
    total_change = (final_price - last_close) / last_close * 100
    direction = "up" if total_change >= 0 else "down"
    summary = (
        f"### Expected price in {HORIZON} trading days: **Rp {final_price:,.0f}**\n\n"
        f"Last close ({last_date.strftime('%Y-%m-%d')}): Rp {last_close:,.0f} → "
        f"projected **{direction} {abs(total_change):.2f}%** by "
        f"{forecast_dates[-1].strftime('%Y-%m-%d')}.\n\n"
        f"_The model was trained on log-transformed prices (to stabilize variance from "
        f"price spikes); forecasts above are exponentiated back to real Rupiah values. "
        f"The shaded band is the model's 10–90% quantile range, not a guarantee._"
    )

    return fig, table, summary


with gr.Blocks(title="BBCA Chronos Forecast", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📈 BBCA Stock Forecast — Fine-tuned Chronos\n"
        f"Forecasts the next **{HORIZON} trading days** of Bank Central Asia (`{TICKER}`) "
        f"closing price using a Chronos model fine-tuned on {CONTEXT}-day context windows "
        f"(`{MODEL_ID}`). Historical prices are pulled live from Yahoo Finance."
    )
    btn = gr.Button("🔁 Refresh forecast", variant="primary")
    summary_box = gr.Markdown()
    plot = gr.Plot()
    table = gr.Dataframe(label=f"Next {HORIZON} trading days", interactive=False)

    btn.click(fn=run_forecast, outputs=[plot, table, summary_box])
    demo.load(fn=run_forecast, outputs=[plot, table, summary_box])

if __name__ == "__main__":
    demo.launch()
