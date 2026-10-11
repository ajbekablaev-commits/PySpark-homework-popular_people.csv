# ================================================
# PySpark Тапсырма: popular_people.csv
# Студент: _______________________
# ================================================
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)


# ---------------- Бөлім 1: SparkSession және деректерді жүктеу ----------------

def create_spark_session():
    """№1. 'PopularPeopleApp' атты SparkSession жасайды."""
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("PopularPeopleApp")
        .getOrCreate()
    )


def load_data(spark, path):
    """№2. CSV жүктеу (header=True, inferSchema=True)."""
    return spark.read.csv(path, header=True, inferSchema=True)


def load_data_with_schema(spark, path):
    """№3. Схеманы StructType арқылы қолмен анықтау (inferSchema=False)."""
    schema = StructType([
        StructField("name", StringType(), True),
        StructField("gender", IntegerType(), True),
        StructField("known_for_department", StringType(), True),
        StructField("original_name", StringType(), True),
        StructField("popularity", DoubleType(), True),
    ])
    return spark.read.csv(path, header=True, schema=schema, inferSchema=False)


# ---------------- Бөлім 2: Деректерді тазалау ----------------

def clean_data(df):
    """№4. '' -> NULL, содан соң known_for_department/name NULL жолдарын жою."""
    df = df.withColumn(
        "known_for_department",
        F.when(F.trim(F.col("known_for_department")) == "", None)
         .otherwise(F.col("known_for_department")),
    )
    return df.dropna(subset=["known_for_department", "name"])


def decode_gender(df):
    """№5. gender (0/1/2/3) -> gender_label."""
    return df.withColumn(
        "gender_label",
        F.when(F.col("gender") == 0, "Unknown")
         .when(F.col("gender") == 1, "Female")
         .when(F.col("gender") == 2, "Male")
         .when(F.col("gender") == 3, "Other")
         .otherwise("Unknown"),
    )


def add_popularity_tier(df):
    """№6. popularity_tier бағанын қосу."""
    return df.withColumn(
        "popularity_tier",
        F.when(F.col("popularity") >= 40, "A-list")
         .when((F.col("popularity") >= 20) & (F.col("popularity") < 40), "B-list")
         .when((F.col("popularity") >= 10) & (F.col("popularity") < 20), "C-list")
         .otherwise("Unknown"),
    )


def normalize_names(df):
    """№7. name бағанын trim + initcap арқылы өңдеу."""
    return df.withColumn("name", F.initcap(F.trim(F.col("name"))))


# ---------------- Бөлім 3: Spark SQL ----------------

def register_temp_view(df):
    """№8. DataFrame-ді 'people' уақытша кестесі ретінде тіркеу."""
    df.createOrReplaceTempView("people")


def dept_stats_sql(spark):
    """№9. Сала бойынша статистика."""
    return spark.sql("""
        SELECT known_for_department,
               COUNT(*)                        AS person_count,
               ROUND(AVG(popularity), 2)       AS avg_popularity,
               MAX(popularity)                 AS max_popularity
        FROM people
        GROUP BY known_for_department
        ORDER BY avg_popularity DESC
    """)


def gender_distribution_sql(spark):
    """№10. gender_label бойынша саны және пайызы."""
    return spark.sql("""
        SELECT gender_label,
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM people
        GROUP BY gender_label
        ORDER BY count DESC
    """)


def top5_by_dept_sql(spark):
    """№11. Әр саладан ең танымал 5 адам (ROW_NUMBER)."""
    return spark.sql("""
        SELECT name, known_for_department, popularity, row_num
        FROM (
            SELECT name, known_for_department, popularity,
                   ROW_NUMBER() OVER (
                       PARTITION BY known_for_department
                       ORDER BY popularity DESC
                   ) AS row_num
            FROM people
        ) t
        WHERE row_num <= 5
        ORDER BY known_for_department, row_num
    """)


# ---------------- Бөлім 4: Партициялау ----------------

def repartition_demo(df):
    """№12. repartition(8), сосын repartition(4, 'known_for_department')."""
    print(f"Бастапқы партиция саны: {df.rdd.getNumPartitions()}")
    df8 = df.repartition(8)
    print(f"repartition(8) кейін: {df8.rdd.getNumPartitions()}")
    df4 = df8.repartition(4, "known_for_department")
    print(f"repartition(4, 'known_for_department') кейін: {df4.rdd.getNumPartitions()}")
    return df4


def coalesce_demo(df):
    """№13. repartition(16), сосын coalesce(4)."""
    wide = df.repartition(16)
    print(f"repartition(16) кейін: {wide.rdd.getNumPartitions()}")
    result = wide.coalesce(4)
    print(f"coalesce(4) кейін: {result.rdd.getNumPartitions()}")
    # coalesce тиімдірек, себебі ол бар партицияларды бір-бірімен біріктіреді
    # және деректерді желі арқылы толық араластыруды (full shuffle) жасамайды.
    # repartition болса барлық деректерді қайта таратады, сондықтан қымбат.
    # coalesce тек партиция санын азайтқанда қолданылады.
    return result


def write_partitioned(df, output_path):
    """№14. Parquet форматында known_for_department бойынша бөліп жазу."""
    (df.write
       .mode("overwrite")
       .partitionBy("known_for_department")
       .parquet(output_path))


def read_single_partition(spark, output_path, dept):
    """№15. Тек бір саланың бумасын оқу."""
    return (spark.read.parquet(output_path)
            .filter(F.col("known_for_department") == dept))


def partition_performance_report(df, base_path="/tmp/partition_perf"):
    """№16. repartition(1) және repartition(8) жазу уақытын салыстыру."""
    t0 = time.time()
    df.repartition(1).write.mode("overwrite").parquet(f"{base_path}/one")
    t1 = time.time() - t0

    t0 = time.time()
    df.repartition(8).write.mode("overwrite").parquet(f"{base_path}/eight")
    t8 = time.time() - t0

    print(f"repartition(1): {t1:.3f} сек")
    print(f"repartition(8): {t8:.3f} сек")
    print(f"Айырмашылық: {abs(t1 - t8):.3f} сек "
          f"({'1' if t1 < t8 else '8'} партиция жылдамырақ)")
    return {"repartition_1": t1, "repartition_8": t8}


# ---------------- Бөлім 5: Жетілдірілген агрегациялар ----------------

def gender_dept_pivot(df):
    """№17. Жолдар - сала, бағандар - gender_label, мән - count."""
    return (df.groupBy("known_for_department")
              .pivot("gender_label")
              .count()
              .fillna(0))


def rank_within_dept(df):
    """№18. Сала ішінде popularity бойынша rank."""
    w = Window.partitionBy("known_for_department").orderBy(F.desc("popularity"))
    return df.withColumn("rank", F.rank().over(w))


def run_full_pipeline(spark, input_path, output_path):
    """№19. Жүктеу -> тазалау -> трансформация -> SQL -> партициялап жазу."""
    df = load_data(spark, input_path)
    df = clean_data(df)
    df = decode_gender(df)
    df = add_popularity_tier(df)
    df = normalize_names(df)

    register_temp_view(df)
    dept_stats_sql(spark).show(truncate=False)
    gender_distribution_sql(spark).show(truncate=False)

    write_partitioned(df, output_path)
    return df
