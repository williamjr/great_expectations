from six import string_types
import os

from ...data_context.types import (
    AllowedKeysDotDict,
)
from . import (
    WriteOnlyStore,
    FilesystemStoreBackend,
    FilesystemStoreBackendConfig,
)

class FileSystemWriteOnlyStoreConfig(AllowedKeysDotDict):
    _allowed_keys = set([
        "base_directory",
    ])
    _required_keys = set([
        "base_directory",
    ])

class FileSystemWriteOnlyStore(WriteOnlyStore):

    config_class = FileSystemWriteOnlyStoreConfig

    def __init__(self, config, root_directory=None):
        assert hasattr(self, 'config_class')

        # NOTE : Abe 2019/09/02 : This is where that config instantiation pattern would be really nice
        self.config = config
        self.root_directory = root_directory
        self._setup()

    def _validate_key(self, key):
        assert isinstance(key, string_types)

    def _validate_value(self, value):
        assert isinstance(value, string_types)

    def _setup(self):
        self.store_backend = self._configure_store_backend({
            "module_name" : "great_expectations.data_context.store",
            "class_name" : "FilesystemStoreBackend",
            "base_directory" : self.config.base_directory,
            "filepath_template" : "{0}", # NOTE : This feels super hacky, but it's what's required to live in a world where store_backends always take tuples as keys
            "replaced_substring" : "xxxxxxxxxxlnsadfl;nasdkgj", # FIXME : This is a dumb hack to not have ANY replaced_substring. This parameter should just be optional.
            "replacement_string" : "____sdajfbaskl;djfbsa__", # FIXME: Ditto this one.
            "key_length" : 1,
        })

    def _set(self, key, value):
        self.store_backend._set( (key,), value )


class DelimitedFileSystemWriteOnlyStoreConfig(AllowedKeysDotDict):
    _allowed_keys = set([
        "base_directory",
        # "delimiter", # Delimiter is defined by the backend
    ])
    _required_keys = set([
        "base_directory",
        # "delimiter",
    ])
    # TODO : Add type checking

class DelimitedFileSystemWriteOnlyStore(WriteOnlyStore):

    config_class = DelimitedFileSystemWriteOnlyStoreConfig

    def __init__(self, config, root_directory=None):
        assert hasattr(self, 'config_class')

        # NOTE : Abe 2019/09/02 : This is where that config instantiation pattern would be really nice
        self.config = config
        self.root_directory = root_directory
        self._setup()

    def _validate_key(self, key):
        assert isinstance(key, tuple)

    def _validate_value(self, value):
        assert isinstance(value, string_types)

    def _setup(self):
        self.store_backend = self._configure_store_backend({
            "module_name" : "great_expectations.data_context.store",
            "class_name" : "DelimitedFilesystemStoreBackend",
            "base_directory" : self.config.base_directory,
        })

    def _set(self, key, value):
        self.store_backend._set( key, value )