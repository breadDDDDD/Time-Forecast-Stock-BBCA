# BBCA Stock Forecast - Fine-tuned Chronos

A lightweight, interactive web application showcasing Amazon's **Chronos** time-series forecasting model, fine-tuned specifically on historical closing prices for **Bank Central Asia (BBCA, IDX)**.

**Training Dataset:** The model was trained on historical daily data spanning from **April 16, 2001, to January 6, 2023**.

**Live Space:** [BBCA-Forecaster on Hugging Face](https://huggingface.co/spaces/SkibidiBreaddd/BBCA-Forecaster)

---

## Features

On application load, the system seamlessly handles the following:
* **Real-time Data Fetching:** Automatically pulls the latest historical closing prices for BBCA.
* **Chronos Inference:** Passes the data through the fine-tuned Chronos forecasting pipeline.
* **Interactive Visualizations:** Renders historical trends alongside future probabilistic prediction intervals.

---

## Tech Stack & Architecture

* **Model Architecture:** Amazon Chronos (Fine-tuned for IDX:BBCA)
* **Dashboard Framework:** Streamlit
* **Data Source:** `yfinance` (or your specific API)
* **Deployment:** Hugging Face Spaces

---

1. Pulls the latest BBCA.JK daily closes from Yahoo Finance.
2. Takes the most recent 512 closes (the context length the model was fine-tuned
   on), log-transforms them, and feeds them to
   [`SkibidiBreaddd/BBCA-Chronos-14062026-v0`](https://huggingface.co/SkibidiBreaddd/BBCA-Chronos-14062026-v0).
3. Plots the 5-trading-day forecast (with an 80% confidence band) against recent
   history, and lists the expected price for each of the next 5 trading days.

Forecasts are produced in log-price space (how the model was trained) and
exponentiated back to Rupiah before display.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploying to Hugging Face Spaces

1. Create a new Space → SDK: **Gradio**.
2. Push `app.py`, `requirements.txt`, and this `README.md` to the Space repo.
3. If `SkibidiBreaddd/BBCA-Chronos-14062026-v0` is a private/gated model, add an
   `HF_TOKEN` secret in the Space settings (Settings → Variables and secrets) —
   the app will pick it up automatically. If it's public, no token is needed.
4. CPU Basic (free tier) is enough to run inference for a model this size.

## Notes / limitations

- "Next 5 trading days" are approximated as the next 5 weekdays — Indonesian
  stock exchange holidays aren't excluded, so dates near a holiday may be off
  by a day.
- This is a demo, not financial advice — see the disclaimer in the app itself.
