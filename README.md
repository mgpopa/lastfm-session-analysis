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
- `top_songs.tsv` - Exercise 2 answer
- `forecast.tsv` - Exercise 3 answer
- `forecast_plot.png` - Visualisation

### 3. Or run locally (needs Python 3.11)

```bash
pip install -r requirements.txt
python exercise2.py
python exercise3.py
python exercise3_validation.py #optional to cross validate the model
```

## Approach

### Exercise 2 - Session setection & top songs

A **session**= consecutive songs by the same user where each song starts within 20 minutes of the previous one. I used PySpark window functions to compute time gaps between plays, flag when the gap exeeds 20 min and assign session IDs via cumulative sum. 

Then I count tracks per session -> take the top 50 -> count song plays across those 50 sessions -> rank -> top 10.

### Exercise 3 - Forecasting

I picked the user with the most sessions, aggregated their daily session count, and used **Prophet** to forecast the next 3 months. Prophet handles weekly/yearly seasonality out of the box, which fits listening behaviour well.

Metric chosen: **number of sessions per day**.

## Assumptions

- A song is identified by `(artist_name, track_name)` since track IDs have many NULLs
- A gap of exactly 20 min = same session; strictly more than 20 min = new session.
- Rows with null user/timestamp/atrist/track are dropped
- Days with no listening activity are counted as 0 sessions for the forecast

## What I'd improve with more time

- Try ARIMA as an alternative to Prophet

    - Tuning Prophet didn't help, ARIMA might be better in production. Prophet is designed for data with trends and multiple seasonality layers, which the user doesn't have. That makes me believe that Prophet might be the wrong choice for this particular user/use case, but it was a reasonable first choice before I had evidence.

- Add feature engineering (day-of-wek effects, holudays etc.)
- Explore session duration as an alternative forecast metric
- More data quality checks upfront
- Better handling of ties in the top 10 ranking
