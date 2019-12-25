import logging
import time

from great_expectations.datasource.types import BatchId
from ..exceptions import BatchKwargsError

from .datasource import Datasource, ReaderMethods
from great_expectations.datasource.generator.subdir_reader_generator import SubdirReaderGenerator
from great_expectations.datasource.generator.databricks_generator import DatabricksTableGenerator
from great_expectations.datasource.generator.in_memory_generator import InMemoryGenerator
from great_expectations.datasource.generator.s3_generator import S3Generator

from great_expectations.types import ClassConfig

logger = logging.getLogger(__name__)

try:
    from great_expectations.dataset.sparkdf_dataset import SparkDFDataset
    from pyspark.sql import SparkSession, DataFrame
except ImportError:
    SparkSession = None
    # TODO: review logging more detail here
    logger.debug("Unable to load pyspark; install optional spark dependency for support.")


class SparkDFDatasource(Datasource):
    """The SparkDFDatasource produces SparkDFDatasets and supports generators capable of interacting with local
    filesystem (the default subdir_reader generator) and databricks notebooks.
    """

    @classmethod
    def build_configuration(cls, data_asset_type=None, generators=None, spark_config=None, **kwargs):
        """
        Build a full configuration object for a datasource, potentially including generators with defaults.

        Args:
            data_asset_type: A ClassConfig dictionary
            generators: Generator configuration dictionary
            spark_config: dictionary of key-value pairs to pass to the spark builder
            **kwargs: Additional kwargs to be part of the datasource constructor's initialization

        Returns:
            A complete datasource configuration.

        """
        if generators is None:
            # Provide a gentle way to build a datasource with a sane default,
            # including ability to specify the base_directory
            base_directory = kwargs.pop("base_directory", "/data")
            reader_options = kwargs.pop("reader_options", {})
            generators = {
                "default": {
                    "class_name": "SubdirReaderGenerator",
                    "base_directory": base_directory,
                    "reader_options": reader_options
                }
            }

        if data_asset_type is None:
            data_asset_type = ClassConfig(
                class_name="SparkDFDataset"
            )
        else:
            try:
                data_asset_type = ClassConfig(**data_asset_type)
            except TypeError:
                # In this case, we allow the passed config, for now, in case they're using a legacy string-only config
                pass

        if spark_config is None:
            spark_config = {}

        configuration = kwargs
        configuration.update({
            "data_asset_type": data_asset_type,
            "generators": generators,
            "spark_config": spark_config
        })
        return configuration

    def __init__(self, name="default", data_context=None, data_asset_type=None, generators=None,
                 spark_config=None, **kwargs):
        """Build a new SparkDFDatasource instance.

        Args:
            name: the name of this datasource
            data_context: the DataContext to which this datasource is connected
            data_asset_type: ClassConfig describing the data_asset type to be constructed by this datasource
            generators: generator configuration
            spark_config: dictionary of key-value pairs to be set on the spark session builder
            **kwargs: Additional
        """
        configuration_with_defaults = SparkDFDatasource.build_configuration(data_asset_type, generators,
                                                                            spark_config, **kwargs)
        data_asset_type = configuration_with_defaults.pop("data_asset_type")
        generators = configuration_with_defaults.pop("generators")
        super(SparkDFDatasource, self).__init__(
            name,
            data_context=data_context,
            data_asset_type=data_asset_type,
            generators=generators,
            **configuration_with_defaults)

        try:
            builder = SparkSession.builder
            for k, v in configuration_with_defaults["spark_config"].items():
                builder.config(k, v)
            self.spark = builder.getOrCreate()
        except AttributeError:
            logger.error("Unable to load spark context; install optional spark dependency for support.")
            self.spark = None

        self._build_generators()

    def _get_generator_class_from_type(self, type_):
        if type_ == "subdir_reader":
            return SubdirReaderGenerator
        elif type_ == "databricks":
            return DatabricksTableGenerator
        elif type_ == "memory":
            return InMemoryGenerator
        elif type_ == "s3":
            return S3Generator
        else:
            raise ValueError("Unrecognized BatchGenerator type %s" % type_)

    def _get_data_asset(self, batch_kwargs, expectation_suite, caching=True, **kwargs):
        """class-private implementation of get_data_asset"""
        if self.spark is None:
            logger.error("No spark session available")
            return None

        for k, v in kwargs.items():
            if isinstance(v, dict):
                if k in batch_kwargs and isinstance(batch_kwargs[k], dict):
                    batch_kwargs[k].update(v)
                else:
                    batch_kwargs[k] = v
            else:
                batch_kwargs[k] = v

        reader_options = batch_kwargs.get("reader_options", {})

        # We need to build a batch_id to be used in the dataframe
        batch_id = BatchId({
            "timestamp": time.time()
        })

        if "data_asset_type" in batch_kwargs:
            data_asset_type_config = reader_options.pop("data_asset_type")  # Get and remove the config
            try:
                data_asset_type_config = ClassConfig(**data_asset_type_config)
            except TypeError:
                # We tried; we'll pass the config downstream, probably as a string, and handle an error later
                pass
        else:
            data_asset_type_config = self._data_asset_type

        data_asset_type = self._get_data_asset_class(data_asset_type_config)

        if not issubclass(data_asset_type, SparkDFDataset):
            raise ValueError("SparkDFDatasource cannot instantiate batch with data_asset_type: '%s'. It "
                             "must be a subclass of SparkDFDataset." % data_asset_type.__name__)

        if "path" in batch_kwargs or "s3" in batch_kwargs:
            # If both are present, let s3 override
            path = batch_kwargs.get("path")
            path = batch_kwargs.get("s3", path)
            reader_method = batch_kwargs.get("reader_method")
            if reader_method is None:
                reader_method = self._guess_reader_method_from_path(path)
                if reader_method is None:
                    raise BatchKwargsError("Unable to determine reader for path: %s" % path, batch_kwargs)
            else:
                try:
                    reader_method = ReaderMethods[reader_method]
                except KeyError:
                    raise BatchKwargsError("Unknown reader method: %s" % reader_method, batch_kwargs)

            reader = self.spark.read

            for option in reader_options.items():
                reader = reader.option(*option)

            if reader_method == ReaderMethods.CSV:
                df = reader.csv(path)
            elif reader_method == ReaderMethods.parquet:
                df = reader.parquet(path)
            elif reader_method == ReaderMethods.delta:
                df = reader.format("delta").load(path)
            else:
                raise BatchKwargsError("Unsupported reader: %s" % reader_method.name, batch_kwargs)

        elif "query" in batch_kwargs:
            df = self.spark.sql(batch_kwargs["query"])

        elif "dataset" in batch_kwargs and isinstance(batch_kwargs["dataset"], (DataFrame, SparkDFDataset)):
            df = batch_kwargs.get("dataset")
            # We don't want to store the actual dataframe in kwargs; copy the remaining batch_kwargs
            batch_kwargs = {k: batch_kwargs[k] for k in batch_kwargs if k != 'dataset'}
            if isinstance(df, SparkDFDataset):
                # Grab just the spark_df reference, since we want to override everything else
                df = df.spark_df
            # Record this in the kwargs *and* the id
            batch_kwargs["SparkDFRef"] = True
            batch_id["SparkDFRef"] = True

        else:
            raise BatchKwargsError("Unrecognized batch_kwargs for spark_source", batch_kwargs)

        if "limit" in batch_kwargs:
            df = df.limit(batch_kwargs['limit'])

        return data_asset_type(df,
                               expectation_suite=expectation_suite,
                               data_context=self._data_context,
                               batch_kwargs=batch_kwargs,
                               caching=caching,
                               batch_id=batch_id)
