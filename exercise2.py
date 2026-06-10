# Top 10 songs played in the top 50 longest sessions by track count
# 1 session = consecutive songs by the same user where each song starts within 20min of the previous one

import glob
import os
import shutil

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, TimestampType

DATA_FILE = "data/lastfm-dataset-1k/userid-timestamp-artid-artname-traid-traname.tsv"
OUTPUT_FILE = "output/top_songs.tsv"
OUTPUT_TMP_DIR = "output/top_songs_tmp"

SESSION_GAP_SECONDS = 20 * 60  # 20 minutes

def main():
    spark = SparkSession.builder.appName("LastFM-Sessions")\
                                .config("spark.driver.memory", "4g")\
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

    # for each user, compute the gap between consecutive plays. if gap >20 min, then new session
    w = Window.partitionBy("user_id").orderBy("timestamp")
    df = df.withColumn("prev_ts", F.lag("timestamp").over(w))
    df = df.withColumn("gap", F.unix_timestamp("timestamp") - F.unix_timestamp("prev_ts"))
    df = df.withColumn("new_session", F.when( (F.col("gap").isNull()) | (F.col("gap") > SESSION_GAP_SECONDS),1,).otherwise(0),)
    df = df.withColumn("session_num", F.sum(F.col("new_session").cast(IntegerType())).over(w))
    df = df.withColumn("session_id", F.concat_ws("_", F.col("user_id"), F.col("session_num").cast(StringType())))

    # top 50 sessions by track record
    session_size = df.groupBy("session_id").agg(F.count("*").alias("track_count"))
    top50 = session_size.orderBy(F.desc("track_count")).limit(50)

    # top 10 songs in those sessions
    tracks_in_top = df.join(top50, "session_id").groupBy("artist_name", "track_name").agg(F.count("*").alias("play_count"))
    rank_w = Window.orderBy(F.desc("play_count"))
    top10 = tracks_in_top.withColumn("rank", F.row_number().over(rank_w)).filter(F.col("rank") <= 10)

    # save as TSV
    result_df = top10.select("rank", "artist_name", "track_name", "play_count")
    result_df.coalesce(1).write.option("sep", "\t").option("header", "true").mode("overwrite").csv(OUTPUT_TMP_DIR)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    part_files = glob.glob(os.path.join(OUTPUT_TMP_DIR, "part-*.csv"))
    if not part_files:
        raise RuntimeError(f"No Spark output file found in {OUTPUT_TMP_DIR}")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    shutil.move(part_files[0], OUTPUT_FILE)
    shutil.rmtree(OUTPUT_TMP_DIR)

    print(f"Top songs saved to {OUTPUT_FILE}")

    spark.stop()

if __name__ == "__main__":
    main()
