# ================================================
# PySpark Тапсырма: popular_people.csv
# Студент: Аблаев Айбек
# ================================================
import os
import shutil
import tempfile
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType


# ---------------- Бөлім 1: SparkSession және жүктеу ----------------

def create_spark_session():
    # master('local[*]') нужен, иначе при запуске через pytest Spark
    # выдаст ошибку "A master URL must be set"
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("PopularPeopleApp")
        .getOrCreate()
    )


def load_data(spark, path):
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("nullValue", "")
        .option("emptyValue", "")
        .csv(path)
    )


def load_data_with_schema(spark, path):
    schema = StructType([
        StructField("name", StringType(), True),
        StructField("gender", IntegerType(), True),
        StructField("known_for_department", StringType(), True),
        StructField("original_name", StringType(), True),
        StructField("popularity", DoubleType(), True),
    ])
    return (
        spark.read.option("header", True)
        .option("nullValue", "")
        .option("emptyValue", "")
        .schema(schema)
        .csv(path)
    )


# ---------------- Бөлім 2: Тазалау ----------------

def clean_data(df):
    # Пустая (или из одних пробелов) строка -> NULL
    df = df.withColumn(
        "known_for_department",
        F.when(F.trim(F.col("known_for_department")) == "", None)
         .otherwise(F.col("known_for_department")),
    )
    df = df.withColumn(
        "name",
        F.when(F.col("name").isNull() | (F.trim(F.col("name")) == ""), None)
         .otherwise(F.trim(F.col("name"))),
    )
    # Удаляем строки, где department или name равны NULL
    return df.dropna(subset=["known_for_department", "name"])


def decode_gender(df):
    gender_value = F.col("gender").cast("int")
    return df.withColumn(
        "gender_label",
        F.when(gender_value == 1, "Female")
         .when(gender_value == 2, "Male")
         .when(gender_value == 3, "Other")
         .otherwise("Unknown"),  # 0 и любые неизвестные/NULL значения
    )


def add_popularity_tier(df):
    popularity = F.col("popularity")
    return df.withColumn(
        "popularity_tier",
        F.when(popularity.isNull(), "Unknown")
         .when(popularity >= 40, "A-list")
         .when(popularity >= 20, "B-list")
         .when(popularity >= 10, "C-list")
         .otherwise("Unknown"),
    )


def normalize_names(df):
    return df.withColumn("name", F.initcap(F.trim(F.col("name"))))


# ---------------- Бөлім 3: Spark SQL ----------------

def register_temp_view(df):
    df.createOrReplaceTempView("people")


def dept_stats_sql(spark):
    return spark.sql("""
        SELECT
            known_for_department,
            COUNT(*)                    AS person_count,
            ROUND(AVG(popularity), 2)   AS avg_popularity,
            MAX(popularity)             AS max_popularity
        FROM people
        GROUP BY known_for_department
        ORDER BY avg_popularity DESC
    """)


def gender_distribution_sql(spark):
    return spark.sql("""
        WITH gender_counts AS (
            SELECT
                gender_label,
                COUNT(*) AS cnt
            FROM people
            GROUP BY gender_label
        )
        SELECT
            gender_label,
            cnt AS `count`,
            ROUND(cnt * 100.0 / SUM(cnt) OVER (), 2) AS percentage
        FROM gender_counts
        ORDER BY cnt DESC
    """)


def top5_by_dept_sql(spark):
    return spark.sql("""
        SELECT *
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY known_for_department
                    ORDER BY popularity DESC
                ) AS rn
            FROM people
        ) t
        WHERE rn <= 5
        ORDER BY known_for_department, rn
    """)


# ---------------- Бөлім 4: Партициялау ----------------

def repartition_demo(df):
    print("Начальное число партиций:", df.rdd.getNumPartitions())

    df8 = df.repartition(8)
    print("После repartition(8):", df8.rdd.getNumPartitions())

    df_dept = df8.repartition(4, "known_for_department")
    print("После repartition(4, 'known_for_department'):",
          df_dept.rdd.getNumPartitions())
    return df_dept


def coalesce_demo(df):
    df16 = df.repartition(16)
    print("После repartition(16):", df16.rdd.getNumPartitions())

    # coalesce эффективнее repartition при уменьшении числа партиций:
    # он не делает полный shuffle, а просто объединяет соседние партиции
    # на тех же исполнителях, поэтому данные почти не пересылаются по сети.
    # repartition же перемешивает все данные между узлами.
    df4 = df16.coalesce(4)
    print("После coalesce(4):", df4.rdd.getNumPartitions())
    return df4


def write_partitioned(df, output_path):
    (
        df.write
        .mode("overwrite")
        .partitionBy("known_for_department")
        .parquet(output_path)
    )


def read_single_partition(spark, output_path, dept):
    # Фильтр по колонке-партиции: Spark читает только нужную папку
    # known_for_department=<dept>/ (partition pruning)
    return (
        spark.read.parquet(output_path)
        .filter(F.col("known_for_department") == dept)
    )


def _timed_write(df, n_partitions):
    tmp_dir = tempfile.mkdtemp(prefix="spark_perf_")
    try:
        start = time.time()
        df.repartition(n_partitions).write.mode("overwrite").parquet(
            os.path.join(tmp_dir, "out")
        )
        return time.time() - start
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def partition_performance_report(df):
    t1 = _timed_write(df, 1)
    t8 = _timed_write(df, 8)

    print(f"repartition(1): {t1:.2f} сек")
    print(f"repartition(8): {t8:.2f} сек")
    if t8 > 0:
        print(f"Отношение времени 1/8 партиций: {t1 / t8:.2f}x")
    return {"repartition_1": t1, "repartition_8": t8}


# ---------------- Бөлім 5: Агрегациялар ----------------

def gender_dept_pivot(df):
    return (
        df.groupBy("known_for_department")
          .pivot("gender_label")
          .count()
          .fillna(0)
    )


def rank_within_dept(df):
    w = Window.partitionBy("known_for_department").orderBy(F.desc("popularity"))
    return df.withColumn("rank", F.rank().over(w))


def run_full_pipeline(spark, input_path, output_path):
    df = load_data(spark, input_path)
    df = clean_data(df)
    df = decode_gender(df)
    df = add_popularity_tier(df)
    df = normalize_names(df)
    register_temp_view(df)
    write_partitioned(df, output_path)
    return df


if __name__ == "__main__":
    spark = create_spark_session()
    result = run_full_pipeline(spark, "data/popular_people.csv", "output/people")
    result.show(5)
    dept_stats_sql(spark).show()
