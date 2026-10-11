# tests/test_homework.py
import pytest
from pyspark.sql import SparkSession
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
from homework import *

DATA_PATH = 'data/popular_people.csv'

@pytest.fixture(scope='session')
def spark():
    return (SparkSession.builder.master('local[2]')
            .appName('TestSuite').getOrCreate())

@pytest.fixture(scope='session')
def raw_df(spark):
    return load_data(spark, DATA_PATH)

@pytest.fixture(scope='session')
def clean_df(raw_df):
    df = clean_data(raw_df)
    df = decode_gender(df)
    df = add_popularity_tier(df)
    return normalize_names(df)

# ---- Тест 1: create_spark_session ----
def test_spark_session():
    s = create_spark_session()
    assert s is not None
    assert s.sparkContext.appName == 'PopularPeopleApp'

# ---- Тест 2: load_data ----
def test_load_data(raw_df):
    assert raw_df.count() > 9000
    assert 'name' in raw_df.columns
    assert 'popularity' in raw_df.columns
    assert 'known_for_department' in raw_df.columns

# ---- Тест 3: load_data_with_schema ----
def test_schema(spark):
    df = load_data_with_schema(spark, DATA_PATH)
    schema_dict = {f.name: f.dataType.typeName() for f in df.schema}
    assert schema_dict['gender'] == 'integer'
    assert schema_dict['popularity'] == 'double'

# ---- Тест 4: clean_data ----
def test_clean_data(clean_df):
    null_count = clean_df.filter(
        clean_df.known_for_department.isNull() | clean_df.name.isNull()
    ).count()
    assert null_count == 0

# ---- Тест 5: decode_gender ----
def test_decode_gender(clean_df):
    assert 'gender_label' in clean_df.columns
    labels = [r.gender_label for r in clean_df.select('gender_label').distinct().collect()]
    for lbl in labels:
        assert lbl in ['Male', 'Female', 'Unknown', 'Other']

# ---- Тест 6: popularity_tier ----
def test_popularity_tier(clean_df):
    assert 'popularity_tier' in clean_df.columns
    tiers = [r.popularity_tier for r in clean_df.select('popularity_tier').distinct().collect()]
    for t in tiers:
        assert t in ['A-list', 'B-list', 'C-list', 'Unknown']

# ---- Тест 7: Spark SQL dept_stats ----
def test_dept_stats_sql(spark, clean_df):
    register_temp_view(clean_df)
    result = dept_stats_sql(spark)
    assert result is not None
    assert 'person_count' in result.columns
    assert 'avg_popularity' in result.columns
    assert result.count() > 0

# ---- Тест 8: repartition ----
def test_repartition(clean_df):
    result = repartition_demo(clean_df)
    assert result is not None
    assert result.rdd.getNumPartitions() > 0

# ---- Тест 9: coalesce ----
def test_coalesce(clean_df):
    result = coalesce_demo(clean_df)
    assert result is not None
    assert result.rdd.getNumPartitions() <= 4

# ---- Тест 10: partitionBy write/read ----
def test_partition_write_read(spark, clean_df, tmp_path):
    out = str(tmp_path / 'output')
    write_partitioned(clean_df, out)
    result = read_single_partition(spark, out, 'Acting')
    assert result.count() > 0
    depts = [r.known_for_department for r in result.select('known_for_department').distinct().collect()]
    assert depts == ['Acting']

# ---- Тест 11: rank_within_dept ----
def test_rank_within_dept(clean_df):
    result = rank_within_dept(clean_df)
    assert 'rank' in result.columns
    # Ең танымал адамның рейтингі = 1 болу тиіс
    top = result.filter(result.rank == 1).count()
    assert top > 0
