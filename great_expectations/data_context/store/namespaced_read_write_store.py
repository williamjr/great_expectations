import logging

import copy
from mimetypes import guess_type
import os

from ..types.base_resource_identifiers import (
    DataContextKey,
)
from great_expectations.data_context.types.resource_identifiers import (
    ExpectationSuiteIdentifier,
    ValidationResultIdentifier,
    SiteSectionIdentifier,
)
from .store import (
    ReadWriteStore,
)
from .store_backend import (
    FixedLengthTupleStoreBackend
)
from great_expectations.data_context.util import (
    load_class,
    instantiate_class_from_config
)
from great_expectations.exceptions import DataContextError
from great_expectations.util import file_relative_path

logger = logging.getLogger(__name__)


class NamespacedReadWriteStore(ReadWriteStore):
    
    def __init__(self,
        store_backend,
        root_directory,
        serialization_type="json"
    ):
        self.store_backend = self._init_store_backend(
            copy.deepcopy(store_backend),
            runtime_config={
                "root_directory": root_directory
            }
        )
        self.root_directory = root_directory
        self.serialization_type = serialization_type

    def _init_store_backend(self, store_backend_config, runtime_config):
        if store_backend_config["class_name"] == "FixedLengthTupleFilesystemStoreBackend":
            config_defaults = {
                "key_length" : 5,
                "module_name" : "great_expectations.data_context.store",
            }
        elif store_backend_config["class_name"] == "FixedLengthTupleS3StoreBackend":
            config_defaults = {
                "key_length": 5, # NOTE: Eugene: 2019-09-06: ???
                "module_name": "great_expectations.data_context.store",
            }
        else:
            config_defaults = {
                "module_name" : "great_expectations.data_context.store",
            }

        return instantiate_class_from_config(
            config=store_backend_config,
            runtime_config=runtime_config,
            config_defaults=config_defaults,
        )

    def _get(self, key):
        self._validate_key(key)

        key_tuple = self._convert_resource_identifier_to_tuple(key)
        return self.store_backend.get(key_tuple)

    def _set(self, key, serialized_value):
        self._validate_key(key)

        key_tuple = self._convert_resource_identifier_to_tuple(key)
        return self.store_backend.set(key_tuple, serialized_value)

    def list_keys(self):
        return [self._convert_tuple_to_resource_identifier(key) for key in self.store_backend.list_keys()]

    def has_key(self, key):
        # NOTE: This is not efficient
        return key in self.list_keys()

    def _convert_resource_identifier_to_tuple(self, key):
        # TODO : Optionally prepend a source_id (the frontend Store name) to the tuple.

        list_ = []
        list_ += self._convert_resource_identifier_to_list(key)

        return tuple(list_)

    def _convert_resource_identifier_to_list(self, key):
        # The logic in this function is recursive, so it can't return a tuple

        list_ = []
        #Fetch keys in _key_order to guarantee tuple ordering in both python 2 and 3
        for key_name in key._key_order:
            key_element = key[key_name]
            if isinstance( key_element, DataContextKey ):
                list_ += self._convert_resource_identifier_to_list(key_element)
            else:
                list_.append(key_element)

        return list_

    def _convert_tuple_to_resource_identifier(self, tuple_):
        new_identifier = self.key_class(*tuple_)
        return new_identifier

    def _validate_key(self, key):
        if not isinstance(key, self.key_class):
            raise TypeError("key: {!r} must be a {}, not {!r}".format(
                key,
                self.key_class,
                type(key),
            ))


class ExpectationsStore(NamespacedReadWriteStore):
    # Note : As of 2019/09/06, this method is untested.
    # It shares virtually all of its business logic with ValidationStore, which is under test.

    def _init_store_backend(self, store_backend_config, runtime_config):
        self.key_class = ExpectationSuiteIdentifier

        store_backend_class_name = store_backend_config.get("class_name", "FixedLengthTupleFilesystemStoreBackend")
        store_backend_module_name = store_backend_config.get("module_name", "great_expectations.data_context.store")
        store_backend_class = load_class(
            class_name=store_backend_class_name,
            module_name=store_backend_module_name
        )
        if issubclass(store_backend_class, FixedLengthTupleStoreBackend):
            config_defaults = {
                "key_length": 4,
                "module_name": "great_expectations.data_context.store",
                "filepath_template": "{0}/{1}/{2}/{3}.json"
            }
        else:
            config_defaults = {
                "module_name": "great_expectations.data_context.store",
            }

        return instantiate_class_from_config(
            config=store_backend_config,
            runtime_config=runtime_config,
            config_defaults=config_defaults,
        )


class ValidationsStore(NamespacedReadWriteStore):
    
    def _init_store_backend(self, store_backend_config, runtime_config):
        self.key_class = ValidationResultIdentifier

        store_backend_class_name = store_backend_config.get("class_name", "FixedLengthTupleFilesystemStoreBackend")
        store_backend_module_name = store_backend_config.get("module_name", "great_expectations.data_context.store")
        store_backend_class = load_class(
            class_name=store_backend_class_name,
            module_name=store_backend_module_name
        )
        if issubclass(store_backend_class, FixedLengthTupleStoreBackend):
            config_defaults = {
                "key_length": 5,
                "module_name": "great_expectations.data_context.store",
                "filepath_template": "{4}/{0}/{1}/{2}/{3}.json"
            }
        else:
            config_defaults = {
                "module_name": "great_expectations.data_context.store",
            }

        return instantiate_class_from_config(
            config=store_backend_config,
            runtime_config=runtime_config,
            config_defaults=config_defaults,
        )


class HtmlSiteStore(NamespacedReadWriteStore):

    def __init__(self,
                 root_directory,
                 serialization_type=None,
                 store_backend=None
                 ):
        self.key_class = SiteSectionIdentifier
        store_backend_module_name = store_backend.get("module_name", "great_expectations.data_context.store")
        store_backend_class_name = store_backend.get("class_name", "FixedLengthTupleFilesystemStoreBackend")
        store_class = load_class(store_backend_class_name, store_backend_module_name)

        if not issubclass(store_class, FixedLengthTupleStoreBackend):
            raise DataContextError("Invalid configuration: HtmlSiteStore needs a FixedLengthTupleStoreBackend")
        if "filepath_template" in store_backend or "key_length" in store_backend:
            logger.warning("Configuring a filepath_template or key_length is not supported in SiteBuilder: "
                           "filepaths will be selected based on the type of asset rendered.")

        # # Each key type gets its own backend.
        # # If backends were DB connections, this could be inefficient, but it doesn't much matter for filepaths.
        # # One thing to watch for is reversibility of keys.
        # # If several types are being writtten to overlapping directories, we could get collisions.
        # expectations_backend_config = copy.deepcopy(store_backend)
        # if "base_directory" in expectations_backend_config:
        #     expectations_backend_config["base_directory"] = os.path.join(expectations_backend_config["base_directory"], "expectations")
        # elif "prefix" in expectations_backend_config:
        #     expectations_backend_config["prefix"] = os.path.join(expectations_backend_config["prefix"], "expectations")
        #
        # validations_backend_config = copy.deepcopy(store_backend)
        # if "base_directory" in validations_backend_config:
        #     validations_backend_config["base_directory"] = os.path.join(validations_backend_config["base_directory"], "validations")
        # elif "prefix" in validations_backend_config:
        #     validations_backend_config["prefix"] = os.path.join(validations_backend_config["prefix"], "validations")

        self.store_backends = {
            ExpectationSuiteIdentifier: instantiate_class_from_config(
                config=store_backend,
                runtime_config={
                    "root_directory": root_directory
                },
                config_defaults={
                    "module_name": "great_expectations.data_context.store",
                    "key_length": 4,
                    "filepath_template": 'expectations/{0}/{1}/{2}/{3}.html',
                }
            ),
            ValidationResultIdentifier: instantiate_class_from_config(
                config=store_backend,
                runtime_config={
                    "root_directory": root_directory
                },
                config_defaults={
                    "module_name": "great_expectations.data_context.store",
                    "key_length": 5,
                    "filepath_template": 'validations/{4}/{0}/{1}/{2}/{3}.html',
                }
            ),
            "index_page":  instantiate_class_from_config(
                config=store_backend,
                runtime_config={
                    "root_directory": root_directory
                },
                config_defaults={
                    "module_name": "great_expectations.data_context.store",
                    "key_length": 0,
                    "filepath_template": 'index.html',
                }
            ),
            "static_assets": instantiate_class_from_config(
                config=store_backend,
                runtime_config={
                    "root_directory": root_directory
                },
                config_defaults={
                    "module_name": "great_expectations.data_context.store",
                    "key_length": None,
                    "filepath_template": None,
                }
            ),
        }

        self.root_directory = root_directory
        self.serialization_type = serialization_type

        # NOTE: Instead of using the filesystem as the source of record for keys,
        # this class trackes keys separately in an internal set.
        # This means that keys are stored for a specific session, but can't be fetched after the original
        # HtmlSiteStore instance leaves scope.
        # Doing it this way allows us to prevent namespace collisions among keys while still having multiple
        # backends that write to the same directory structure.
        # It's a pretty reasonable way for HtmlSiteStore to do its job---you just have to remember that it
        # can't necessarily set and list_keys like most other Stores.
        self.keys = set()

    def _convert_tuple_to_resource_identifier(self, tuple_):
        if tuple_[0] == "expectations":
            resource_identifier = ExpectationSuiteIdentifier(*tuple_[1])
        elif tuple_[0] == "validations":
            resource_identifier = ValidationResultIdentifier(*tuple_[1])
        else:
            raise Exception("unknown section name: " + tuple_[0])
        new_identifier = SiteSectionIdentifier(site_section_name=tuple_[0], resource_identifier=resource_identifier)
        return new_identifier

    def _get(self, key):
        self._validate_key(key)

        key_tuple = self._convert_resource_identifier_to_tuple(key.resource_identifier)
        return self.store_backends[
            type(key.resource_identifier)
        ].get(key_tuple)

    def _set(self, key, serialized_value):
        self._validate_key(key)

        self.keys.add(key)

        key_tuple = self._convert_resource_identifier_to_tuple(key.resource_identifier)
        return self.store_backends[
            type(key.resource_identifier)
        ].set(key_tuple, serialized_value, content_encoding='utf-8', content_type='text/html; charset=utf-8')

    def _validate_key(self, key):
        if not isinstance(key, SiteSectionIdentifier):
            raise TypeError("key: {!r} must a SiteSectionIdentifier, not {!r}".format(
                key,
                type(key),
            ))

        for key_class in self.store_backends.keys():
            try:
                if isinstance(key.resource_identifier, key_class):
                    return
            except TypeError:
                # it's ok to have a key that is not a type (e.g. the string "index_page")
                continue

        # The key's resource_identifier didn't match any known key_class
        raise TypeError("resource_identifier in key: {!r} must one of {}, not {!r}".format(
            key,
            set(self.store_backends.keys()),
            type(key),
        ))

    def list_keys(self):
        return [self._convert_tuple_to_resource_identifier(("expectations", key)) for key in self.store_backends[ExpectationSuiteIdentifier].list_keys()] + \
               [self._convert_tuple_to_resource_identifier(("validations", key)) for key in self.store_backends[ValidationResultIdentifier].list_keys()]

    def write_index_page(self, page):
        """This third store has a special method, which uses a zero-length tuple as a key."""
        return self.store_backends["index_page"].set((), page, content_encoding='utf-8', content_type='text/html; '
                                                                                                      'charset=utf-8')
    
    def copy_static_assets(self, static_assets_source_dir=None):
        """
        Copies static assets, using a special "static_assets" backend store that accepts variable-length tuples as
        keys, with no filepath_template.
        """
        file_exclusions = [".DS_Store"]
        dir_exclusions = []
        
        if not static_assets_source_dir:
            static_assets_source_dir = file_relative_path(__file__, "../../render/view/static")

        for item in os.listdir(static_assets_source_dir):
            # Directory
            if os.path.isdir(os.path.join(static_assets_source_dir, item)):
                if item in dir_exclusions:
                    continue
                # Recurse
                new_source_dir = os.path.join(static_assets_source_dir, item)
                self.copy_static_assets(new_source_dir)
            # File
            else:
                # Copy file over using static assets store backend
                if item in file_exclusions:
                    continue
                source_name = os.path.join(static_assets_source_dir, item)
                with open(source_name, 'rb') as f:
                    # Only use path elements starting from static/ for key
                    store_key = tuple(os.path.normpath(source_name).split(os.sep))
                    store_key = store_key[store_key.index('static'):]
                    content_type, content_encoding = guess_type(item, strict=False)
                    self.store_backends["static_assets"].set(
                        store_key,
                        f.read(),
                        content_encoding=content_encoding,
                        content_type=content_type
                    )
