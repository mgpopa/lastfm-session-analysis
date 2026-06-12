# Forecast the number of sessions per day for the user with the most sessions
# use Prophet to forecast the next 3 months from the last available record
# reuse the same sessionization logic from exercise 2

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt
from prophet import Prophet
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, TimestampType

DATA_FILE = "data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
OUTPUT_FILE = "output/forecast.tsv"
PLOT_FILE = "output/forecast_plot.png"

SESSION_GAP_SECONDS = 20 * 60  # 20 minutes
FORECAST_DAYS = 90

def sessionize(df):
    w = Window.partitionBy("user_id").orderBy("timestamp")
    df = df.withColumn("prev_ts", F.lag("timestamp").over(w))
    df = df.withColumn("gap", F.unix_timestamp("timestamp") - F.unix_timestamp("prev_ts"))
    df = df.withColumn("new_session", F.when( (F.col("gap").isNull()) | (F.col("gap") > SESSION_GAP_SECONDS),1,).otherwise(0),)
    df = df.withColumn("session_num", F.sum(F.col("new_session").cast(IntegerType())).over(w))
    df = df.withColumn("session_id", F.concat_ws("_", F.col("user_id"), F.col("session_num").cast(StringType())))
    return df

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

    spark.stop()


    # train Prophet model | LE: initially yearly_seasonality was set to true and seasonality_mode="multiplicative"
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1
    )
    model.fit(daily)

    future = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    forecast = model.predict(future)

    # extract forecast period only
    last_date = daily["ds"].max()
    fc = forecast[forecast["ds"] > last_date][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fc.columns = ["date", "predicted_sessions", "lower_bound", "upper_bound"]
    fc["predicted_sessions"] = fc["predicted_sessions"].clip(lower=0).round(1)
    fc["lower_bound"] = fc["lower_bound"].clip(lower=0).round(1)
    fc["upper_bound"] = fc["upper_bound"].round(1)
    fc["date"] = fc["date"].dt.strftime("%Y-%m-%d")

    fc.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Forecast saved to {OUTPUT_FILE}")
    print(fc.head(10).to_string(index=False))

    # plot forecast
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["ds"], daily["y"], "k.", alpha=0.3, markersize=5, label="Historical")
    ax.plot(forecast["ds"], forecast["yhat"], "b-", linewidth=1, label="Forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="blue", alpha=0.2, label="Confidence Interval")
    ax.axvline(x=last_date, color="red", linestyle="--", label="Forecast Start")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Sessions/Day")
    ax.legend()
    ax.set_title(f"Forecast of Daily Sessions for User {top_user_id}")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()
    print(f"Forecast plot saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
