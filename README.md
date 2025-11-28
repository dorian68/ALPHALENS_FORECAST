# AlphaLens Forecast

Hybrid forecasting framework that combines Prophet, NeuralProphet, NHITS, TFT, and EGARCH volatility modelling with Monte Carlo simulation to build AlphaLens AI trading setups.

## Features
- Automated mean-model selection based on timeframe (<=30m -> NHITS, 30m-4h -> NeuralProphet, >=4h -> Prophet).
- Optional TFT forecaster for extended experimentation with sequence-to-sequence deep learning.
- EGARCH(1,1) volatility modelling with Student-t innovations for tail-aware risk.
- Monte Carlo engine delivering TP/SL hit probabilities and price quantiles.
- Risk engine generates direction, TP/SL, risk/reward, confidence, and position sizing.

## Project Structure
```
alphalens_forecast/
├── core/
├── data/
│   └── provider.py
├── forecasting.py
├── models/
│   ├── router.py
│   └── selection.py
├── training.py
├── training_schedule.py
├── utils/
├── config.py
├── main.py
├── requirements.txt
└── README.md

config/
└── instruments.yml
```

## Prerequisites
- Python 3.9 or newer
- pip/venv (or conda) for dependency management

## Installation
```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Architecture Overview
- **DataProvider** (`alphalens_forecast/data/provider.py`) fetches OHLCV via Twelve Data and keeps a local cache (default `data/cache`) so repeated forecasts do not re-download the entire history. Override the cache path with `--data-cache-dir`.
- **ModelRouter** (`alphalens_forecast/models/router.py`) owns the filesystem layout `models/{model_type}/{symbol}/{timeframe}/model.pkl`, ensuring every (instrument, timeframe) pair has its own artifact (EGARCH included).
- **ForecastEngine** (`alphalens_forecast/forecasting.py`) performs inference end-to-end: load data, choose NHITS/NeuralProphet/Prophet based on the timeframe, load EGARCH, run Monte Carlo, and build the AlphaLens payload.
- **Training entrypoints** (`alphalens_forecast/training.py`) expose `train_nhits`, `train_neuralprophet`, `train_prophet`, and `train_egarch` so desks can retrain offline (AWS Batch/Cron) and simply let the CLI handle inference.

## Training Cadence & Instrument Universe
- Official cadences live in `alphalens_forecast/training_schedule.py` (NHITS every 2–3 days, NeuralProphet/Prophet weekly, EGARCH daily).
- Instruments/timeframes are listed in `config/instruments.yml` (G10 FX crosses + BTC/ETH on 15m/1h/4h) along with the default horizons. Modify this file to extend the universe or point to external storage (e.g., S3 sync).

## Configuration
Copy `.env.example` (create one if needed) or set environment variables directly. Supported keys:

| Variable | Description | Default |
|----------|-------------|---------|
| `TWELVE_DATA_API_KEY` | Twelve Data API key | none |
| `DEFAULT_SYMBOL` | Default symbol for CLI | `BTC/USD` |
| `DEFAULT_INTERVAL` | Default timeframe | `15min` |
| `DATA_OUTPUT_SIZE` | Max bars to request | `5000` |
| `MC_PATHS` | Monte Carlo paths | `3000` |
| `MC_SEED` | RNG seed (optional) | unset |
| `USE_MONTECARLO` | Enable Monte Carlo | `true` |
| `TARGET_ANNUAL_VOL` | Target annual volatility for sizing | `0.20` |
| `CONFIDENCE_THRESHOLD` | Signal confidence threshold | `0.60` |

The application loads `.env` automatically via `python-dotenv`.

## Usage
```bash
python -m alphalens_forecast.main --symbol BTC/USD --timeframe 15min --horizons 3 6 12 24
```

Optional flags:
- `--paths 5000` to adjust Monte Carlo paths.
- `--no-montecarlo` to skip simulation (quantile fallback is used).
- `--output forecasts.json` to save the JSON payload.
- `--log-level DEBUG` for verbose logging.
- `--data-cache-dir data/cache_fx` to relocate the OHLCV cache used by `DataProvider`.
- `--model-dir /mnt/artifacts/models` to control where ModelRouter stores `models/{model_type}/{symbol}/{timeframe}`.
- `--save-models --reuse-model` to keep the legacy CLI manifests (in addition to the router-managed assets) and reuse them when the input hash matches.

The CLI prints a JSON payload per the AlphaLens schema, including median price, TP/SL, prob-hit TP before SL, and risk metrics per horizon.

### Offline Training
```python
python - <<'PY'
from alphalens_forecast.training import train_nhits, train_neuralprophet, train_prophet, train_egarch

symbol = "EUR/USD"
for timeframe in ("15min", "1h", "4h"):
    train_nhits(symbol, timeframe)
    train_neuralprophet(symbol, timeframe)
    train_prophet(symbol, timeframe)
    train_egarch(symbol, timeframe)
PY
```
These helpers load data through the `DataProvider`, fit the selected model, and persist it via `ModelRouter`. Schedule them according to `training_schedule.py` so the CLI only needs to perform inference.

## Monte Carlo Settings
Monte Carlo draws Student-t innovations scaled by EGARCH volatility forecasts. Each simulation evaluates whether the take-profit level is reached before the stop-loss and records distribution quantiles (`p20`, `p50`, `p80`). Adjust `MC_PATHS` and `MC_SEED` to balance runtime and stability.

## Development Notes
- Heavy dependencies (Prophet, Torch, Darts) may require system libraries (ujson, pystan build tools).
- GPU acceleration is optional; set `TORCH_DEVICE` if desired.
- Logging uses `tqdm` for progress bars when Monte Carlo is active.

## License
MIT (customise as needed).
