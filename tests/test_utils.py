from __future__ import division

import pytest
import locale
from functools import wraps
import random
import string
import copy
from collections import (
    OrderedDict,
    Mapping
)

from dateutil.parser import parse

import pandas as pd
import numpy as np

from great_expectations.dataset import PandasDataset, SqlAlchemyDataset, SparkDFDataset
from great_expectations.profile import ColumnsExistProfiler

try:
    import sqlalchemy.dialects.sqlite as sqlitetypes
    SQLITE_TYPES = {
        "VARCHAR": sqlitetypes.VARCHAR,
        "CHAR": sqlitetypes.CHAR,
        "INTEGER": sqlitetypes.INTEGER,
        "SMALLINT": sqlitetypes.SMALLINT,
        "DATETIME": sqlitetypes.DATETIME(truncate_microseconds=True),
        "DATE": sqlitetypes.DATE,
        "FLOAT": sqlitetypes.FLOAT,
        "BOOLEAN": sqlitetypes.BOOLEAN
    }
except ImportError:
    SQLITE_TYPES = {}

try:
    import sqlalchemy.dialects.postgresql as postgresqltypes
    POSTGRESQL_TYPES = {
        "TEXT": postgresqltypes.TEXT,
        "CHAR": postgresqltypes.CHAR,
        "INTEGER": postgresqltypes.INTEGER,
        "SMALLINT": postgresqltypes.SMALLINT,
        "BIGINT": postgresqltypes.BIGINT,
        "TIMESTAMP": postgresqltypes.TIMESTAMP,
        "DATE": postgresqltypes.DATE,
        "DOUBLE_PRECISION": postgresqltypes.DOUBLE_PRECISION,
        "BOOLEAN": postgresqltypes.BOOLEAN,
        "NUMERIC": postgresqltypes.NUMERIC
    }
except ImportError:
    POSTGRESQL_TYPES = {}

try:
    import sqlalchemy.dialects.mysql as mysqltypes
    MYSQL_TYPES = {
        "TEXT": mysqltypes.TEXT,
        "CHAR": mysqltypes.CHAR,
        "INTEGER": mysqltypes.INTEGER,
        "SMALLINT": mysqltypes.SMALLINT,
        "BIGINT": mysqltypes.BIGINT,
        "TIMESTAMP": mysqltypes.TIMESTAMP,
        "DATE": mysqltypes.DATE,
        "FLOAT": mysqltypes.FLOAT,
        "BOOLEAN": mysqltypes.BOOLEAN
    }
except ImportError:
    MYSQL_TYPES = {}


def modify_locale(func):
    @wraps(func)
    def locale_wrapper(*args, **kwargs):
        old_locale = locale.setlocale(locale.LC_TIME, None)
        print(old_locale)
        # old_locale = locale.getlocale(locale.LC_TIME) Why not getlocale? not sure
        try:
            new_locale = locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
            assert new_locale == 'en_US.UTF-8'
            func(*args, **kwargs)
        except Exception:
            raise
        finally:
            locale.setlocale(locale.LC_TIME, old_locale)

    return locale_wrapper


# Taken from the following stackoverflow:
# https://stackoverflow.com/questions/23549419/assert-that-two-dictionaries-are-almost-equal
def assertDeepAlmostEqual(expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using pyteset.approx and checks all other values with an assertion equality statement
    Accepts additional positional and keyword arguments and pass those
    intact to pytest.approx() (that's how you specify comparison
    precision).

    """
    is_root = '__trace' not in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        # if isinstance(expected, (int, float, long, complex)):
        if isinstance(expected, (int, float, complex)):
            assert expected == pytest.approx(actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            assert len(expected) == len(actual)
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assertDeepAlmostEqual(v1, v2,
                                      __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            assert set(expected) == set(actual)
            for key in expected:
                assertDeepAlmostEqual(expected[key], actual[key],
                                      __trace=repr(key), *args, **kwargs)
        else:
            assert expected == actual
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (str(exc), trace))
        raise exc


def get_dataset(dataset_type, data, schemas=None, profiler=ColumnsExistProfiler, caching=True):
    """Utility to create datasets for json-formatted tests.
    """
    df = pd.DataFrame(data)
    if dataset_type == 'PandasDataset':
        if schemas and "pandas" in schemas:
            schema = schemas["pandas"]
            pandas_schema = {}
            for (key, value) in schema.items():
                # Note, these are just names used in our internal schemas to build datasets *for internal tests*
                # Further, some changes in pandas internal about how datetimes are created means to support pandas
                # pre- 0.25, we need to explicitly specify when we want timezone.

                # We will use timestamp for timezone-aware (UTC only) dates in our tests
                if value.lower() in ["timestamp", "datetime64[ns, tz]"]:
                    df[key] = pd.to_datetime(df[key], utc=True)
                    continue
                elif value.lower() in ["datetime", "datetime64", "datetime64[ns]"]:
                    df[key] = pd.to_datetime(df[key])
                    continue
                try:
                    type_ = np.dtype(value)
                except TypeError:
                    type_ = getattr(pd.core.dtypes.dtypes, value)
                    # If this raises AttributeError it's okay: it means someone built a bad test
                pandas_schema[key] = type_
            # pandas_schema = {key: np.dtype(value) for (key, value) in schemas["pandas"].items()}
            df = df.astype(pandas_schema)
        return PandasDataset(df, profiler=profiler, caching=caching)

    elif dataset_type == "sqlite":
        from sqlalchemy import create_engine
        engine = create_engine('sqlite://')
        conn = engine.connect()
        # Add the data to the database as a new table

        sql_dtypes = {}
        if schemas and "sqlite" in schemas and isinstance(engine.dialect, sqlitetypes.dialect):
            schema = schemas["sqlite"]
            sql_dtypes = {col: SQLITE_TYPES[dtype] for (col,dtype) in schema.items()}
            for col in schema:
                type_ = schema[col]
                if type_ in ["INTEGER", "SMALLINT", "BIGINT"]:
                    df[col] = pd.to_numeric(df[col], downcast='signed')
                elif type_ in ["FLOAT", "DOUBLE", "DOUBLE_PRECISION"]:
                    df[col] = pd.to_numeric(df[col])
                elif type_ in ["DATETIME", "TIMESTAMP"]:
                    df[col] = pd.to_datetime(df[col])

        tablename = "test_data_" + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])
        df.to_sql(name=tablename, con=conn, index=False, dtype=sql_dtypes)

        # Build a SqlAlchemyDataset using that database
        return SqlAlchemyDataset(tablename, engine=conn, profiler=profiler, caching=caching)

    elif dataset_type == 'postgresql':
        from sqlalchemy import create_engine
        # Create a new database
        engine = create_engine('postgresql://postgres@localhost/test_ci')
        conn = engine.connect()

        sql_dtypes = {}
        if schemas and "postgresql" in schemas and isinstance(engine.dialect, postgresqltypes.dialect):
            schema = schemas["postgresql"]
            sql_dtypes = {col: POSTGRESQL_TYPES[dtype] for (col, dtype) in schema.items()}
            for col in schema:
                type_ = schema[col]
                if type_ in ["INTEGER", "SMALLINT", "BIGINT"]:
                    df[col] = pd.to_numeric(df[col], downcast='signed')
                elif type_ in ["FLOAT", "DOUBLE", "DOUBLE_PRECISION"]:
                    df[col] = pd.to_numeric(df[col])
                elif type_ in ["DATETIME", "TIMESTAMP"]:
                    df[col] = pd.to_datetime(df[col])

        tablename = "test_data_" + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])
        df.to_sql(name=tablename, con=conn, index=False, dtype=sql_dtypes)

        # Build a SqlAlchemyDataset using that database
        return SqlAlchemyDataset(tablename, engine=conn, profiler=profiler, caching=caching)

    elif dataset_type == 'mysql':
        engine = create_engine('mysql://root@localhost/test_ci')
        conn = engine.connect()

        sql_dtypes = {}
        if schemas and "mysql" in schemas and isinstance(engine.dialect, mysqltypes.dialect):
            schema = schemas["mysql"]
            sql_dtypes = {col: MYSQL_TYPES[dtype] for (col, dtype) in schema.items()}
            for col in schema:
                type_ = schema[col]
                if type_ in ["INTEGER", "SMALLINT", "BIGINT"]:
                    df[col] = pd.to_numeric(df[col], downcast='signed')
                elif type_ in ["FLOAT", "DOUBLE", "DOUBLE_PRECISION"]:
                    df[col] = pd.to_numeric(df[col])
                elif type_ in ["DATETIME", "TIMESTAMP"]:
                    df[col] = pd.to_datetime(df[col])

        tablename = "test_data_" + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])
        df.to_sql(name=tablename, con=conn, index=False, dtype=sql_dtypes)

        # Build a SqlAlchemyDataset using that database
        return SqlAlchemyDataset(tablename, engine=conn, profiler=profiler, caching=caching)

    elif dataset_type == 'SparkDFDataset':
        from pyspark.sql import SparkSession
        import pyspark.sql.types as sparktypes

        SPARK_TYPES = {
            "StringType": sparktypes.StringType,
            "IntegerType": sparktypes.IntegerType,
            "LongType": sparktypes.LongType,
            "DateType": sparktypes.DateType,
            "TimestampType": sparktypes.TimestampType,
            "FloatType": sparktypes.FloatType,
            "DoubleType": sparktypes.DoubleType,
            "BooleanType": sparktypes.BooleanType,
            "DataType": sparktypes.DataType,
            "NullType": sparktypes.NullType
        }


        spark = SparkSession.builder.getOrCreate()
        # We need to allow null values in some column types that do not support them natively, so we skip
        # use of df in this case.
        data_reshaped = list(zip(*[v for _, v in data.items()]))  # create a list of rows
        if schemas and 'spark' in schemas:
            schema = schemas['spark']
            # sometimes first method causes Spark to throw a TypeError
            try:
                spark_schema = sparktypes.StructType([
                    sparktypes.StructField(column, SPARK_TYPES[schema[column]](), True)
                    for column in schema
                ])
                # We create these every time, which is painful for testing
                # However nuance around null treatment as well as the desire
                # for real datetime support in tests makes this necessary
                data = copy.deepcopy(data)
                if "ts" in data:
                    print(data)
                    print(schema)
                for col in schema:
                    type_ = schema[col]
                    if type_ in ["IntegerType", "LongType"]:
                        # Ints cannot be None...but None can be valid in Spark (as Null)
                        vals = []
                        for val in data[col]:
                            if val is None:
                                vals.append(val)
                            else:
                                vals.append(int(val))
                        data[col] = vals
                    elif type_ in ["FloatType", "DoubleType"]:
                        vals = []
                        for val in data[col]:
                            if val is None:
                                vals.append(val)
                            else:
                                vals.append(float(val))
                        data[col] = vals
                    elif type_ in ["DateType", "TimestampType"]:
                        vals = []
                        for val in data[col]:
                            if val is None:
                                vals.append(val)
                            else:
                                vals.append(parse(val))
                        data[col] = vals
                # Do this again, now that we have done type conversion using the provided schema
                data_reshaped = list(zip(*[v for _, v in data.items()]))  # create a list of rows
                spark_df = spark.createDataFrame(data_reshaped, schema=spark_schema)
            except TypeError:
                string_schema = sparktypes.StructType([
                    sparktypes.StructField(column, sparktypes.StringType())
                    for column in schema
                ])
                spark_df = spark.createDataFrame(data_reshaped, string_schema)
                for c in spark_df.columns:
                    spark_df = spark_df.withColumn(c, spark_df[c].cast(SPARK_TYPES[schema[c]]()))
        elif len(data_reshaped) == 0:
            # if we have an empty dataset and no schema, need to assign an arbitrary type
            columns = list(data.keys())
            spark_schema = sparktypes.StructType([
                sparktypes.StructField(column, sparktypes.StringType())
                for column in columns
            ])
            spark_df = spark.createDataFrame(data_reshaped, spark_schema)
        else:
            # if no schema provided, uses Spark's schema inference
            columns = list(data.keys())
            spark_df = spark.createDataFrame(data_reshaped, columns)
        return SparkDFDataset(spark_df, profiler=profiler, caching=caching)

    else:
        raise ValueError("Unknown dataset_type " + str(dataset_type))


def candidate_getter_is_on_temporary_notimplemented_list(context, getter):
    if context in ["sqlite"]:
        return getter in [
            'get_column_modes',
            'get_column_stdev'
        ]
    if context in ["postgresql", "mysql"]:
        return getter in [
            'get_column_modes'
        ]
    if context == 'SparkDFDataset':
        return getter in []


def candidate_test_is_on_temporary_notimplemented_list(context, expectation_type):
    if context in ["sqlite", "postgresql", "mysql"]:
        return expectation_type in [
            # "expect_column_to_exist",
            # "expect_table_row_count_to_be_between",
            # "expect_table_row_count_to_equal",
            # "expect_table_columns_to_match_ordered_list",
            # "expect_column_values_to_be_unique",
            # "expect_column_values_to_not_be_null",
            # "expect_column_values_to_be_null",
            # "expect_column_values_to_be_of_type",
            # "expect_column_values_to_be_in_type_list",
            # "expect_column_values_to_be_in_set",
            # "expect_column_values_to_not_be_in_set",
            # "expect_column_distinct_values_to_be_in_set",
            # "expect_column_distinct_values_to_equal_set",
            # "expect_column_distinct_values_to_contain_set",
            # "expect_column_values_to_be_between",
            "expect_column_values_to_be_increasing",
            "expect_column_values_to_be_decreasing",
            # "expect_column_value_lengths_to_be_between",
            # "expect_column_value_lengths_to_equal",
            # "expect_column_values_to_match_regex",
            # "expect_column_values_to_not_match_regex",
            # "expect_column_values_to_match_regex_list",
            # "expect_column_values_to_not_match_regex_list",
            "expect_column_values_to_match_strftime_format",
            "expect_column_values_to_be_dateutil_parseable",
            "expect_column_values_to_be_json_parseable",
            "expect_column_values_to_match_json_schema",
            # "expect_column_mean_to_be_between",
            # "expect_column_median_to_be_between",
            # "expect_column_quantile_values_to_be_between",
            "expect_column_stdev_to_be_between",
            #"expect_column_unique_value_count_to_be_between",
            #"expect_column_proportion_of_unique_values_to_be_between",
            "expect_column_most_common_value_to_be_in_set",
            # "expect_column_sum_to_be_between",
            # "expect_column_min_to_be_between",
            # "expect_column_max_to_be_between",
            # "expect_column_chisquare_test_p_value_to_be_greater_than",
            "expect_column_bootstrapped_ks_test_p_value_to_be_greater_than",
            # "expect_column_kl_divergence_to_be_less_than",
            "expect_column_parameterized_distribution_ks_test_p_value_to_be_greater_than",
            "expect_column_pair_values_to_be_equal",
            "expect_column_pair_values_A_to_be_greater_than_B",
            "expect_column_pair_values_to_be_in_set",
            "expect_multicolumn_values_to_be_unique"
        ]
    if context == "SparkDFDataset":
        return expectation_type in [
            # "expect_column_to_exist",
            # "expect_table_row_count_to_be_between",
            # "expect_table_row_count_to_equal",
            # "expect_table_columns_to_match_ordered_list",
            # "expect_column_values_to_be_unique",
            # "expect_column_values_to_not_be_null",
            # "expect_column_values_to_be_null",
            # "expect_column_values_to_be_of_type",
            # "expect_column_values_to_be_in_type_list",
            # "expect_column_values_to_be_in_set",
            # "expect_column_values_to_not_be_in_set",
            # "expect_column_distinct_values_to_be_in_set",
            # "expect_column_distinct_values_to_equal_set",
            # "expect_column_distinct_values_to_contain_set",
            # "expect_column_values_to_be_between",
            "expect_column_values_to_be_increasing",
            "expect_column_values_to_be_decreasing",
            # "expect_column_value_lengths_to_be_between",
            # "expect_column_value_lengths_to_equal",
            # "expect_column_values_to_match_regex",
            # "expect_column_values_to_not_match_regex",
            "expect_column_values_to_match_regex_list",
            "expect_column_values_to_not_match_regex_list",
            # "expect_column_values_to_match_strftime_format",
            "expect_column_values_to_be_dateutil_parseable",
            "expect_column_values_to_be_json_parseable",
            "expect_column_values_to_match_json_schema",
            # "expect_column_mean_to_be_between",
            # "expect_column_median_to_be_between",            
            # "expect_column_quantile_values_to_be_between",
            # "expect_column_stdev_to_be_between",
            # "expect_column_unique_value_count_to_be_between",
            # "expect_column_proportion_of_unique_values_to_be_between",
            # "expect_column_most_common_value_to_be_in_set",
            # "expect_column_sum_to_be_between",
            # "expect_column_min_to_be_between",
            # "expect_column_max_to_be_between",
            # "expect_column_chisquare_test_p_value_to_be_greater_than",
            "expect_column_bootstrapped_ks_test_p_value_to_be_greater_than",
            # "expect_column_kl_divergence_to_be_less_than",
            "expect_column_parameterized_distribution_ks_test_p_value_to_be_greater_than",
            "expect_column_pair_values_to_be_equal",
            "expect_column_pair_values_A_to_be_greater_than_B",
            "expect_column_pair_values_to_be_in_set",
            "expect_multicolumn_values_to_be_unique"
        ]
    return False


def evaluate_json_test(data_asset, expectation_type, test):
    """
    This method will evaluate the result of a test build using the Great Expectations json test format.

    NOTE: Tests can be suppressed for certain data types if the test contains the Key 'suppress_test_for' with a list
        of DataAsset types to suppress, such as ['SQLAlchemy', 'Pandas'].

    :param data_asset: (DataAsset) A great expectations DataAsset
    :param expectation_type: (string) the name of the expectation to be run using the test input
    :param test: (dict) a dictionary containing information for the test to be run. The dictionary must include:
        - title: (string) the name of the test
        - exact_match_out: (boolean) If true, match the 'out' dictionary exactly against the result of the expectation
        - in: (dict or list) a dictionary of keyword arguments to use to evaluate the expectation or a list of positional arguments
        - out: (dict) the dictionary keys against which to make assertions. Unless exact_match_out is true, keys must\
            come from the following list:
              - success
              - observed_value
              - unexpected_index_list
              - unexpected_list
              - details
              - traceback_substring (if present, the string value will be expected as a substring of the exception_traceback)
    :return: None. asserts correctness of results.
    """

    data_asset.set_default_expectation_argument('result_format', 'COMPLETE')

    if 'title' not in test:
        raise ValueError(
            "Invalid test configuration detected: 'title' is required.")

    if 'exact_match_out' not in test:
        raise ValueError(
            "Invalid test configuration detected: 'exact_match_out' is required.")

    if 'in' not in test:
        raise ValueError(
            "Invalid test configuration detected: 'in' is required.")

    if 'out' not in test:
        raise ValueError(
            "Invalid test configuration detected: 'out' is required.")

    # Support tests with positional arguments
    if isinstance(test['in'], list):
        result = getattr(data_asset, expectation_type)(*test['in'])
    # As well as keyword arguments
    else:
        result = getattr(data_asset, expectation_type)(**test['in'])

    # Check results
    if test['exact_match_out'] is True:
        assert test['out'] == result
    else:
        for key, value in test['out'].items():
            # Apply our great expectations-specific test logic

            if key == 'success':
                assert result['success'] == value

            elif key == 'observed_value':
                if 'tolerance' in test:
                    if isinstance(value, dict):
                        assert set(value.keys()) == set(result["result"]["observed_value"].keys())
                        for k,v in value.items():
                            assert np.allclose(result["result"]["observed_value"][k], v, rtol=test["tolerance"])
                    else:
                        assert np.allclose(result['result']['observed_value'], value, rtol=test['tolerance'])
                else:
                    assert value == result['result']['observed_value']

            # NOTE: This is a key used ONLY for testing cases where an expectation is legitimately allowed to return
            # any of multiple possible observed_values. expect_column_values_to_be_of_type is one such expectation.
            elif key == "observed_value_list":
                assert result["result"]["observed_value"] in value

            elif key == 'unexpected_index_list':
                if isinstance(data_asset, (SqlAlchemyDataset, SparkDFDataset)):
                    pass
                else:
                    assert result['result']['unexpected_index_list'] == value

            elif key == 'unexpected_list':
                assert result['result']['unexpected_list'] == value, "expected " + \
                    str(value) + " but got " + \
                    str(result['result']['unexpected_list'])

            elif key == 'details':
                assert result['result']['details'] == value

            elif key == "value_counts":
                for val_count in value:
                    assert val_count in result["result"]["details"]["value_counts"]

            elif key.startswith("observed_cdf"):
                if "x_-1" in key:
                    if key.endswith("gt"):
                        assert result["result"]["details"]["observed_cdf"]["x"][-1] > value
                    else:
                        assert result["result"]["details"]["observed_cdf"]["x"][-1] == value
                elif "x_0" in key:
                    if key.endswith("lt"):
                        assert result["result"]["details"]["observed_cdf"]["x"][0] < value
                    else:
                        assert result["result"]["details"]["observed_cdf"]["x"][0] == value
                else:
                    raise ValueError(
                        "Invalid test specification: unknown key " + key + " in 'out'")

            elif key == 'traceback_substring':
                assert result['exception_info']['raised_exception']
                assert value in result['exception_info']['exception_traceback'], "expected to find " + \
                    value + " in " + \
                    result['exception_info']['exception_traceback']
            
            elif key == "expected_partition":
                assert np.allclose(result["result"]["details"]["expected_partition"]["bins"], value["bins"])
                assert np.allclose(result["result"]["details"]["expected_partition"]["weights"], value["weights"])
                if "tail_weights" in result["result"]["details"]["expected_partition"]:
                    assert np.allclose(result["result"]["details"]["expected_partition"]["tail_weights"], value["tail_weights"])
     
            elif key == "observed_partition":
                assert np.allclose(result["result"]["details"]["observed_partition"]["bins"], value["bins"])
                assert np.allclose(result["result"]["details"]["observed_partition"]["weights"], value["weights"])
                if "tail_weights" in result["result"]["details"]["observed_partition"]:
                    assert np.allclose(result["result"]["details"]["observed_partition"]["tail_weights"], value["tail_weights"])

            else:
                raise ValueError(
                    "Invalid test specification: unknown key " + key + " in 'out'")


def dict_to_ordered_dict(plain_dict):
    ordered_dict = OrderedDict()
    for key, val in plain_dict.items():
        if isinstance(val, Mapping):
            ordered_dict[key] = dict_to_ordered_dict(val)
        else:
            ordered_dict[key] = val
    return ordered_dict
