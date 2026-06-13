# ARIMA comparison, tried auto_arima against the naive baseline

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pmdarima as pm
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, TimestampType

DATA_FILE = "data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
OUTPUT_FILE = "output/arima_forecast.tsv"
OUTPUT_VALIDATION_FILE = "output/arima_validation.tsv"
PLOT_FILE = "output/arima_plot.png"

SESSION_GAP_SECONDS = 20 * 60  # 20 minutes
HORIZON = 30
FORECAST_DAYS = 90

####### reused from exercise3 (same data loading and sessionalization)

def sessionize(df):
    w = Window.partitionBy("user_id").orderBy("timestamp")
    df = df.withColumn("prev_ts", F.lag("timestamp").over(w))
    df = df.withColumn("gap", F.unix_timestamp("timestamp") - F.unix_timestamp("prev_ts"))
    df = df.withColumn("new_session", F.when( (F.col("gap").isNull()) | (F.col("gap") > SESSION_GAP_SECONDS),1,).otherwise(0),)
    df = df.withColumn("session_num", F.sum(F.col("new_session").cast(IntegerType())).over(w))
    df = df.withColumn("session_id", F.concat_ws("_", F.col("user_id"), F.col("session_num").cast(StringType())))
    return df

def build_daily_series(spark):

    # load the TSV
    schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("artist_id", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("track_id", StringType(), True),
        StructField("track_name", StringType(), True)
    ])
    df = spark.read.csv(DATA_FILE, sep="\t", header=False, schema=schema)
    df = df.withColumn("timestamp", F.col("timestamp").cast(TimestampType()))
    df = df.filter(F.col("user_id").isNotNull() & F.col("timestamp").isNotNull() & F.col("artist_name").isNotNull() & F.col("track_name").isNotNull())
    df = sessionize(df)

    # find user with most sessions
    user_sessions = df.select("user_id", "session_id").distinct().groupBy("user_id").agg(F.count("*").alias("count"))
    top_user = user_sessions.orderBy(F.desc("count")).first()
    top_user_id = top_user["user_id"]
    print(f"Top user: {top_user_id} ({top_user['count']} sessions)")

    # build daily session count for that user
    user_df = df.filter(F.col("user_id") == top_user_id)
    session_starts = user_df.groupBy("session_id").agg(F.min("timestamp").alias("ts"))
    session_starts = session_starts.withColumn("date", F.to_date("ts"))
    daily = session_starts.groupBy("date").count().toPandas()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds")

    # fill missing days with 0 sessions
    full_range = pd.date_range(start=daily["ds"].min(), end=daily["ds"].max())
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).rename_axis("ds").reset_index()
    daily["y"] = daily["y"].astype(int)

    return daily, top_user_id

###### end of reused code from exercise3. New validation logic below


def main():
    spark = SparkSession.builder.appName("LastFM-Forecast")\
                                .master("local[2]")\
                                .config("spark.driver.memory", "4g")\
                                .config("spark.sql.shuffle.partitions", "32")\
                                .config("spark.default.parallelism", "4")\
                                .config("spark.sql.adaptive.enabled", "true")\
                                .config("spark.ui.enabled", "false")\
                                .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    daily, top_user_id = build_daily_series(spark)
    spark.stop()

    print(f"Top user {top_user_id}, {len(daily)} days of data")

    #hold out last 30 days for testing
    train = daily.iloc[:-HORIZON]
    test = daily.iloc[-HORIZON:]
    actual=test["y"].values

    # auto_arima to find best (p,d,q) automatically, m=7 for weekly pattern
    print("Fitting ARIMA")
    model= pm.auto_arima(train["y"], seasonal=True, m=7, suppress_warnings=True, stepwise=True)
    print(f"Selected order: {model.order}, seasonal: {model.seasonal_order}")

    pred = model.predict(n_periods=HORIZON)
    mae = (abs(actual - pred)).mean()
    print(f"ARIMA MAE: {mae:.2f} sessions/day")

    #save validation results
    val_out = pd.DataFrame({"date": test["ds"].dt.strftime('%Y-%m-%d'), "actual": actual, "predicted": pred.round(1)})
    val_out.to_csv(OUTPUT_VALIDATION_FILE, index=False, sep="\t")

    # retrain on all data, forecast next 90 days
    full_model = pm.auto_arima(daily["y"], seasonal=True, m=7, suppress_warnings=True, stepwise=True)
    full_pred = full_model.predict(n_periods=FORECAST_DAYS).clip(lower=0) #no negative forecasts as sessions can't be negative

    last_date = daily["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
    fc_out= pd.DataFrame({"date": future_dates.strftime('%Y-%m-%d'), "predicted_sessions": full_pred.round(1)})
    fc_out.to_csv(OUTPUT_FILE, index=False, sep="\t")
    print(f"Forecast saved to {OUTPUT_FILE}")

    #plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["ds"], daily["y"], "k.",alpha=0.3, markersize=2, label="Historical")
    ax.plot(future_dates, full_pred, "r-", linewidth=1.5, label=f"ARIMA Forecast")
    ax.axvline(x=last_date, color="gray", linestyle="--", label="Forecast Start")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sessions/day")
    ax.set_title(f"ARIMA 90-days forecast {top_user_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"Plot saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()