import os

from ..util import (
    safe_mmkdir
)
from .store_backend import (
    StoreBackendConfig,
    StoreBackend,
)

class DelimitedFilesystemStoreBackendConfig(StoreBackendConfig):
    _allowed_keys = set([
        "base_directory"
    ])
    _required_keys = _allowed_keys

class DelimitedFilesystemStoreBackend(StoreBackend):
    """Uses a local filepath as a store.
    """

    config_class = DelimitedFilesystemStoreBackendConfig

    def __init__(
        self,
        config,
        root_directory, #This argument is REQUIRED for this class
    ):
        """
        Q: What was the original rationale for keeping root_directory out of config?
        A: Because it's passed in separately to the DataContext. If we want the config to be serializable to yaml, we can't add extra arguments at runtime.

        HOWEVER, passing in root_directory as a separate parameter breaks the normal pattern we've been using for configurability.

        TODO: Figure this out. It might require adding root_directory to the data_context config...?
        NOTE: One possibility is to add a `runtime_config` parallel to the existing `config` in all our configurable classes.
        Then root_directory can be an element within the runtime_config.
        """

        if not os.path.isabs(root_directory):
            raise ValueError("root_directory must be an absolute path. Got {0} instead.".format(root_directory))
            
        self.root_directory = root_directory
        
        super(DelimitedFilesystemStoreBackend, self).__init__(config)


    def _setup(self):
        self.full_base_directory = os.path.join(
            self.root_directory,
            self.config.base_directory,
        )

        # TODO : Consider re-implementing this:
        # safe_mmkdir(str(os.path.dirname(self.full_base_directory)))

    # NOTE : This is identical to FilesystemStoreBackend
    def _get(self, key):
        filepath = os.path.join(
            self.full_base_directory,
            self._convert_key_to_filepath(key)
        )
        with open(filepath) as infile:
            return infile.read()

    # NOTE : This is identical to FilesystemStoreBackend
    def _set(self, key, value):
        filepath = os.path.join(
            self.full_base_directory,
            self._convert_key_to_filepath(key)
        )
        path, filename = os.path.split(filepath)

        safe_mmkdir(str(path))
        with open(filepath, "w") as outfile:
            outfile.write(value)

    def _validate_key(self, key):
        super(DelimitedFilesystemStoreBackend, self)._validate_key(key)
        
    # NOTE : This is identical to FilesystemStoreBackend
    def list_keys(self):
        # TODO : Rename "keys" in this method to filepaths, for clarity
        key_list = []
        for root, dirs, files in os.walk(self.full_base_directory):
            for file_ in files:
                full_path, file_name = os.path.split(os.path.join(root, file_))
                relative_path = os.path.relpath(
                    full_path,
                    self.full_base_directory,
                )
                if relative_path == ".":
                    key = file_name
                else:
                    key = os.path.join(
                        relative_path,
                        file_name
                    )

                key_list.append(
                    self._convert_filepath_to_key(key)
                )

        return key_list

    # TODO : Write tests for this method
    def has_key(self, key):
        assert isinstance(key, string_types)

        all_keys = self.list_keys()
        return key in all_keys

    def _convert_key_to_filepath(self, key):
        self._validate_key(key)

        converted_string = str(os.path.join(*key))
        return converted_string

    def _convert_filepath_to_key(self, filepath):
        path = os.path.normpath(filepath)
        return tuple(filepath.split(os.sep))
