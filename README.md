# Stock Price Prediction with News Sentiment & LSTM

*Does what people say about the world help predict tomorrow's stock price? An end-to-end machine-learning project on 8 years of Dow Jones data.*

---

## What is this?

This project asks a simple question: **if you read every major news headline each day, can you predict whether the stock market will go up or down tomorrow?**

To answer it, we combined two ideas:

1. **News sentiment** — for every day from August 2008 to July 2016, we took the top 25 news headlines and scored each one as negative, neutral, or positive using a pre-trained language model.
2. **Price history + technical indicators** — we paired those daily sentiment scores with the Dow Jones Industrial Average (DJIA) opening, high, low, and closing prices, plus a handful of standard market indicators that traders have used for decades.

Then we fed all of that into an **LSTM neural network** — a type of model designed to learn patterns in sequences over time — and asked it to predict the next day's closing price.

**The headline result:** our best model beats a "naive" baseline (which just guesses that tomorrow's price will equal today's). Removing sentiment from the inputs makes the model worse, which is evidence that **news sentiment carries real predictive signal**, even if it's modest.

### Results at a glance

| # | Model | MASE ↓ | RMSE ↓ |
|---|---|---:|---:|
| 1 | Base LSTM | 1.593 | 116.12 |
| 2 | Deep LSTM with Fine Tuning | 1.439 | 116.11 |
| 3 | Fine Tuned Base LSTM | 1.067 | 116.08 |
| 4 | Fine Tuned LSTM + Extra Features | 1.035 | 116.08 |
| 5 | **Fine Tuned LSTM + More Features** | **0.981** | **116.17** |
| 6 | Fine Tuned LSTM *without* Sentiment | 1.030 | 116.17 |

> **MASE < 1.0 means we beat the naive baseline.** Only Model 5 manages it. Removing sentiment (Model 6) takes us back above 1.0 — sentiment helps.

---

## Visual preview

The notebooks generate the following plots — open them after running to see the full picture:

- `model.ipynb` cell 19 / 25 / 29 / 43 / 50 / 56 — predicted vs actual closing price for each model on the last 150 days
- `model.ipynb` cell 60 — bar charts comparing MASE / MSE / RMSE across all six models
- `prep.ipynb` cell 31 / 32 — DJIA closing-price trend and yearly distribution
- `prep.ipynb` cell 33 — Dickey-Fuller stationarity test, autocorrelation plots
- `prep.ipynb` cells 36 / 39 / 43 / 47 — correlation heatmaps between sentiment, prices, and engineered features

---

## Quick Start (≈5 minutes)

This path uses pre-computed sentiment scores cached in `dataset/final_df.csv`. No huge model download, no waiting.

```bash
# 1. Clone the repo
git clone git@github.com:elvinsug/stock-analysis.git
cd stock-analysis

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Then in the browser:

1. Open **`model.ipynb`**.
2. Run all cells top to bottom (`Kernel → Restart & Run All`).
3. Watch the models train and the plots appear.

**Why this works without running `prep.ipynb` first:** the file `dataset/final_df.csv` already contains the merged price + sentiment dataset, so `model.ipynb` just loads it and skips straight to modelling.

---

## Full Reproduction (≈1 hour)

Use this path if you want to regenerate the sentiment scores from scratch — for example, to verify the pipeline or experiment with a different language model.

> **Heads up:** the first time you run `prep.ipynb`, it will download the [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) model (~500 MB). Sentiment scoring runs on **49,725 headlines** (1,989 days × 25 per day). Expect ~30–60 minutes on CPU, much faster on GPU.

1. Run **`prep.ipynb`** end to end. This will:
   - Load the raw news + price data from `dataset/`
   - Clean the byte-string artefacts in the headlines
   - Score every headline for sentiment
   - Save the merged result back to `dataset/final_df.csv`
2. Then run **`model.ipynb`** as in Quick Start.

---

## How it works (zero-background explanations)

### Sentiment analysis
We use a pre-trained language model (RoBERTa, fine-tuned on Twitter sentiment) that reads each news headline and outputs three numbers: how *negative*, *neutral*, and *positive* it sounds. Think of it as a robot newspaper editor that gives every headline a mood score.

### LSTM (Long Short-Term Memory)
A type of neural network designed to learn patterns in sequences. Unlike a regular network that sees inputs as a "bag" of independent values, an LSTM remembers what came before — making it well-suited to time-series data where today depends on yesterday.

### Naive baseline
The simplest possible "prediction": *tomorrow's price = today's price*. It sounds silly, but for short-term stock prices it's surprisingly hard to beat. Every model in this project is judged against it.

### MASE (Mean Absolute Scaled Error)
A score that compares our model's average error to the naive baseline's average error.
- **MASE = 1.0** → we're as good as the naive guess.
- **MASE < 1.0** → we're better.
- **MASE > 1.0** → we're worse (the naive guess wins).

### Technical indicators (used as extra features)
Standard tools traders have used for decades. The notebooks compute and explain each one, but for orientation:

| Indicator | One-line intuition |
|---|---|
| **EMA** (Exponential Moving Average) | A smoothed average that gives more weight to recent prices |
| **MACD** | Difference between a short and long EMA — flags trend changes |
| **RSI** (Relative Strength Index) | Whether the stock has been bought or sold *too aggressively* recently |
| **SMA** (Simple Moving Average) | A plain rolling average over a window of days |
| **ATR** (Average True Range) | How much the price typically swings each day — a volatility measure |
| **Stochastic Oscillator** | Where today's price sits relative to its recent range |
| **Bollinger Bands** | Upper and lower "envelope" bands around a moving average |

---

## Project structure

```
stock-analysis/
├── prep.ipynb              # Data preprocessing, sentiment scoring, EDA
├── model.ipynb             # LSTM model design, training, evaluation
├── requirements.txt        # Python dependencies (pinned with floors)
├── pyrightconfig.json      # Static type-checking config
├── README.md               # You are here
└── dataset/
    ├── Combined_News_DJIA.csv     # Raw: top 25 daily headlines + up/down label
    ├── upload_DJIA_table.csv      # Raw: daily DJIA OHLC + volume
    ├── sentiment.csv              # Cached: 75 sentiment features per day
    └── final_df.csv               # Cached: merged prices + sentiment, model-ready
```

---

## Results & findings

- **Best model:** *Fine Tuned LSTM + More Features* (Model 5), with **MASE = 0.981** — meaning our average prediction error is ~2% smaller than the naive baseline's.
- **Sentiment matters:** stripping sentiment from the same model (Model 6) pushes MASE back to 1.030. Modest but real signal.
- **Optimal lookback window:** 9 days. We tested 1–20 — 9 minimised both MASE and MSE.
- **Linear models would not work here.** Correlation analysis showed that no single feature has a strong linear relationship with the closing price; the model has to learn non-linear interactions, which is exactly what LSTMs are good at.
- **L2 regularization (0.0013)** and **He-normal initialization** noticeably improved generalisation.

---

## Limitations & honest caveats

- **Single train/test split.** We used the last ~150 days as a fixed validation set rather than k-fold cross-validation, so results are vulnerable to one specific time window's quirks.
- **Manual hyperparameter tuning.** No grid / Bayesian search — chosen values reflect intuition and a handful of trial runs.
- **Data ends in July 2016.** Models have not been validated against post-2016 market regimes (zero-rate era, COVID, 2022 inflation shock, etc.).
- **General news, not finance-specific.** The sentiment model was trained on Twitter and is being applied to general-news headlines. A finance-domain sentiment model would likely produce a stronger signal.
- **Modest improvement.** MASE of 0.981 is statistically a win but not a tradeable edge — this is a learning project, not a strategy.

---

## Skills demonstrated

This project was originally a university assignment, but I rebuilt and documented it to practise the kind of work that financial-services infrastructure and data teams actually do.

- **Data pipelines & reproducibility** — deterministic preprocessing, intermediate datasets cached to disk, fixed random seeds (`reset_seed()`), separate "build the data" and "use the data" notebooks so heavy steps are not re-run unnecessarily.
- **Software engineering hygiene** — virtual environment, pinned `requirements.txt`, static type-checking config (`pyrightconfig.json`), notebooks structured so each cell has one clear purpose.
- **Quantitative analysis** — exploratory data analysis with correlation heatmaps, stationarity testing (Dickey-Fuller), domain-aware feature engineering using standard financial indicators (EMA, MACD, RSI, ATR, Bollinger Bands, Stochastic Oscillator).
- **Machine learning** — neural network architecture design, regularization (L2), weight initialisation, learning-rate scheduling, early stopping, systematic model iteration with each step justified against a baseline.
- **Communication** — every modelling decision is explained in plain English in the notebook, results are framed against a baseline so a non-expert can interpret them, and the README provides both a beginner path and a full reproduction path.

---

## Tech stack

| Library | Role |
|---|---|
| `pandas`, `numpy` | Data wrangling and numerical computing |
| `transformers`, `torch` | Pre-trained sentiment-analysis model (HuggingFace) |
| `tensorflow`, `keras` | LSTM model definition, training, callbacks |
| `scikit-learn` | `StandardScaler` for feature normalisation |
| `statsmodels` | Stationarity tests and time-series diagnostics |
| `scipy` | `softmax` for sentiment-score normalisation |
| `matplotlib`, `seaborn` | All plots and heatmaps |
| `tqdm` | Progress bars during long loops |
| `jupyter`, `ipywidgets` | Notebook environment |

See [`requirements.txt`](requirements.txt) for the exact version floors.

---

## Author

Elvin Sugianto

---

## Acknowledgements

- Dataset: [*Daily News for Stock Market Prediction*](https://www.kaggle.com/datasets/aaron7sun/stocknews) by Aaron7sun on Kaggle, which itself draws on Reddit WorldNews and Yahoo Finance.
- Sentiment model: [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) by the Cardiff NLP group.
- Originally developed for **IE0005 — Introduction to Data Science and Artificial Intelligence** at NTU Singapore.

---

## License

Released under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` if added.
