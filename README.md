# LastFM Session Analysis

## How to run

### 1. Get the data

Download from http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html and extract into `data/`:

```bash
mkdir -p data && cd data
wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
tar -xzf lastfm-dataset-1K.tar.gz
cd .. 
```

### 2. Run with Docker

```bash
docker-compose up --build
```

Results will appear in `output/`:

**Exercise 2 :**
- `top_songs.tsv` (top 10 songs in tje top 50 longest sessions)

**Exercise 3 - Prophet forecast:**
- `forecast.tsv` (90 days forecast)
- `forecast_plot.png` (historical data & Prophet forecast)

**Exercise 3 - model validation:**
- `validation_results.tsv` (Prophet vs naive baseline MAE comparison)
- `valudation_plot.png` (cross-validation predictions vs actuals)

**Exercise 3 - ARIMA alternative:**
- `arima_forecast.tsv` (90 days forecast, ARIMA better MAE than Prophet)
- `arima_validation.png` (ARIMA 30 days holdout: predicted vs actual)
- `arima_plot.png` (historical data + ARIMA forecast)

### 3. Or run locally (needs Python 3.11)

```bash
pip install -r requirements.txt
python exercise2.py
python exercise3.py
python exercise3_validation.py #optional to cross validate the model
python exercise3_arima.py #optional to try ARIMA as alternative
```

## Approach

### Exercise 2 - Session setection & top songs

A **session**= consecutive songs by the same user where each song starts within 20 minutes of the previous one. I used PySpark window functions to compute time gaps between plays, flag when the gap exeeds 20 min and assign session IDs via cumulative sum. 

Then I count tracks per session -> take the top 50 -> count song plays across those 50 sessions -> rank -> top 10.

### Exercise 3 - Forecasting

Metric chosen: **number of sessions per day**.

I picked the user with the most sessions, aggregated their daily session count, and started with **Prophet** since it handles weekly/yearly seasonality out of the box (`exercise3.py`).

Then I wanted to check if Prophet actually adds value, so I ran cross-validation against a naive baseline (predict= avg for that weekday). Turns out the baseline wins: MAE = 1.69 vs Prophet = 1.94. The user's behaviour is very regular by day-of-week and Prophet's extra complexity doesn't help (`exercise3_validation.py`).

That made me to try **ARIMA** as a simpler alternative (auto_arima) with weekly seasonality (`exercise3_arima.py`). It focuses on short-term patterns (yesterday predicts today) rather than decomposing into trend + seasonality.

ARIMA scored MAE = 1.33, best of the three. It captures both the weekly pattern and short-term momentum (yesterday predicts today). So the ARIMA script also produces the full 90 days forecast as an alternative to Prophet's.


## Assumptions

- A song is identified by `(artist_name, track_name)` since track IDs have many NULLs
- A gap of exactly 20 min = same session; strictly more than 20 min = new session.
- Rows with null user/timestamp/atrist/track are dropped
- Days with no listening activity are counted as 0 sessions for the forecast

## What I'd improve with more time

- Tune ARIMA further (e.g. try different seasonal periods)
- Add feature engineering (day-of-wek effects, holidays etc.)
- Explore session duration as an alternative forecast metric
- More data quality checks upfront
- Pull the shared logic out into a small `helpers.py` that both `exercise3.py` and `exercise3_validation.py` import

### Initial improvement ideas (now implemented)
- ~~Compare Prophet vs ARIMA~~
- ~~Try ARIMA as an alternative to Prophet~~

    ~~- Tuning Prophet didn't help, ARIMA might be better in production. Prophet is designed for data with trends and multiple seasonality layers, which the user doesn't have. That makes me believe that Prophet might be the wrong choice for this particular user/use case, but it was a reasonable first choice before I had evidence.~~

- ~~Do proper cross-validation~~