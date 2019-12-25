import random

from ..types import (
    DataAssetIdentifier,
    ValidationResultIdentifier,
)
from ..types.base_resource_identifiers import (
    DataContextKey,
)
from ..util import safe_mmkdir
import pandas as pd
import six
import io
import os
import json
import logging
logger = logging.getLogger(__name__)
import importlib
import re
from six import string_types

from ..util import (
    parse_string_to_data_context_resource_identifier
)
from ...types import (
    ListOf,
    AllowedKeysDotDict,
)

# TODO : Add docstrings to these classes.
# TODO : Implement S3StoreBackend with mocks and tests

# NOTE : Abe 2019/08/30 : Currently, these classes behave as key-value stores.
# We almost certainly want to extend that functionality to allow other operations

class StoreBackend(object):
    """a key-value store, to abstract away reading and writing to a persistence layer
    """

    def __init__(
        self,
        root_directory=None, # NOTE: Eugene: 2019-09-06: I think this should be moved into filesystem-specific children classes
    ):
        self.root_directory = root_directory

    def get(self, key):
        self._validate_key(key)
        value = self._get(key)
        return value

    def set(self, key, value, **kwargs):
        self._validate_key(key)
        self._validate_value(value)
        # Allow the implementing setter to return something (e.g. a path used for its key)
        return self._set(key, value, **kwargs)

    def has_key(self, key):
        self._validate_key(key)
        return self._has_key(key)

    def _validate_key(self, key):
        if not isinstance(key, tuple):
            raise TypeError("Keys in {0} must be instances of {1}, not {2}".format(
                self.__class__.__name__,
                tuple,
                type(key),
            ))
        
        for key_element in key:
            if not isinstance(key_element, string_types):
                raise TypeError("Elements within tuples passed as keys to {0} must be instances of {1}, not {2}".format(
                    self.__class__.__name__,
                    string_types,
                    type(key_element),
                ))

    def _validate_value(self, value):
        pass

    def _get(self, key):
        raise NotImplementedError

    def _set(self, key, value, **kwargs):
        raise NotImplementedError

    def list_keys(self):
        raise NotImplementedError

    def _has_key(self, key):
        raise NotImplementedError


class InMemoryStoreBackend(StoreBackend):
    """Uses an in-memory dictionary as a store backend.

    Note: currently, this class turns the whole key into a single key_string.
    This works, but it's blunt.
    """

    def __init__(
        self,
        separator=".",
        root_directory=None
    ):
        self.store = {}
        self.separator = separator

    def _get(self, key):
        return self.store[self._convert_tuple_to_string(key)]

    def _set(self, key, value, **kwargs):
        self.store[self._convert_tuple_to_string(key)] = value

    def _validate_key(self, key):
        super(InMemoryStoreBackend, self)._validate_key(key)
        
        if self.separator in key:
            raise ValueError("Keys in {0} must not contain the separator character {1} : {2}".format(
                self.__class__.__name__,
                self.separator,
                key,
            ))
    
    def _convert_tuple_to_string(self, tuple_):
        return self.separator.join(tuple_)
    
    def _convert_string_to_tuple(self, string):
        return tuple(string.split(self.separator))

    def list_keys(self):
        return [self._convert_string_to_tuple(key_str) for key_str in list(self.store.keys())]

    def _has_key(self, key):
        return self._convert_tuple_to_string(key) in self.store


class FixedLengthTupleStoreBackend(StoreBackend):
    """

    The key to this StoreBackend abstract class must be a tuple with fixed length equal to key_length.
    The filepath_template is a string template used to convert the key to a filepath.
    There's a bit of regex magic in _convert_filepath_to_key that reverses this process,
    so that we can write AND read using filenames as keys.

    Another class should get this logic through multiple inheritance.
    """

    def __init__(
        self,
        # base_directory,
        filepath_template,
        key_length,
        root_directory,
        forbidden_substrings=None,
        platform_specific_separator=True
    ):
        assert isinstance(key_length, int)
        self.key_length = key_length
        if forbidden_substrings is None:
            forbidden_substrings = ["/", "\\"]
        self.forbidden_substrings = forbidden_substrings
        self.platform_specific_separator = platform_specific_separator

        self.filepath_template = filepath_template
        self.verify_that_key_to_filepath_operation_is_reversible()

    def _validate_key(self, key):
        super(FixedLengthTupleStoreBackend, self)._validate_key(key)

        for key_element in key:
            for substring in self.forbidden_substrings:
                if substring in key_element:
                    raise ValueError("Keys in {0} must not contain substrings in {1} : {2}".format(
                        self.__class__.__name__,
                        self.forbidden_substrings,
                        key,
                    ))

    def _validate_value(self, value):
        # NOTE: We may want to allow bytes here as well.

        if not isinstance(value, string_types):
            raise TypeError("Values in {0} must be instances of {1}, not {2}".format(
                self.__class__.__name__,
                string_types,
                type(value),
            ))

    def _convert_key_to_filepath(self, key):
        # NOTE: At some point in the future, it might be better to replace this logic with os.path.join.
        # That seems more correct, but the configs will be a lot less intuitive.
        # In the meantime, there is some chance that configs will not be cross-OS compatible.

        # NOTE : These methods support fixed-length keys, but not variable.
        self._validate_key(key)
        converted_string = self.filepath_template.format(*list(key))
        if self.platform_specific_separator:
            converted_string = os.path.join(*converted_string.split('/'))
        return converted_string

    def _convert_filepath_to_key(self, filepath):
        # filepath_template (for now) is always specified with forward slashes, but it is then
        # used to (1) dynamically construct and evaluate a regex, and (2) split the provided (observed) filepath
        if self.platform_specific_separator:
            filepath_template = os.path.join(*self.filepath_template.split('/'))
            filepath_template = filepath_template.replace('\\', '\\\\')
        else:
            filepath_template = self.filepath_template

        # Convert the template to a regex
        indexed_string_substitutions = re.findall(r"{\d+}", filepath_template)
        tuple_index_list = ["(?P<tuple_index_{0}>.*)".format(i, ) for i in range(len(indexed_string_substitutions))]
        intermediate_filepath_regex = re.sub(
            r"{\d+}",
            lambda m, r=iter(tuple_index_list): next(r),
            filepath_template
        )
        filepath_regex = intermediate_filepath_regex.format(*tuple_index_list)

        # Apply the regex to the filepath
        matches = re.compile(filepath_regex).match(filepath)
        if matches is None:
            return None

        # Map key elements into the appropriate parts of the tuple
        new_key = list([None for element in range(self.key_length)])
        for i in range(len(tuple_index_list)):
            tuple_index = int(re.search('\d+', indexed_string_substitutions[i]).group(0))
            key_element = matches.group('tuple_index_' + str(i))
            new_key[tuple_index] = key_element

        new_key = tuple(new_key)
        return new_key

    def verify_that_key_to_filepath_operation_is_reversible(self):
        def get_random_hex(len=4):
            return "".join([random.choice(list("ABCDEF0123456789")) for i in range(len)])

        key = tuple([get_random_hex() for j in range(self.key_length)])
        filepath = self._convert_key_to_filepath(key)
        new_key = self._convert_filepath_to_key(filepath)
        if key != new_key:
            raise ValueError(
                "filepath template {0} for class {1} is not reversible for a tuple of length {2}. Have you included all elements in the key tuple?".format(
                    self.filepath_template,
                    self.__class__.__name__,
                    self.key_length,
                ))


class FixedLengthTupleFilesystemStoreBackend(FixedLengthTupleStoreBackend):
    """Uses a local filepath as a store.

    The key to this StoreBackend must be a tuple with fixed length equal to key_length.
    The filepath_template is a string template used to convert the key to a filepath.
    There's a bit of regex magic in _convert_filepath_to_key that reverses this process,
    so that we can write AND read using filenames as keys.
    """

    def __init__(
        self,
        base_directory,
        filepath_template,
        key_length,
        root_directory,
        forbidden_substrings=None,
        platform_specific_separator=True
    ):
        super(FixedLengthTupleFilesystemStoreBackend, self).__init__(
            root_directory=root_directory,
            filepath_template=filepath_template,
            key_length=key_length,
            forbidden_substrings=forbidden_substrings,
            platform_specific_separator=platform_specific_separator
        )

        self.base_directory = base_directory

        if not os.path.isabs(root_directory):
            raise ValueError("root_directory must be an absolute path. Got {0} instead.".format(root_directory))

        self.root_directory = root_directory

        self.full_base_directory = os.path.join(
            self.root_directory,
            self.base_directory,
        )

        safe_mmkdir(str(os.path.dirname(self.full_base_directory)))

    def _get(self, key):
        filepath = os.path.join(
            self.full_base_directory,
            self._convert_key_to_filepath(key)
        )
        with open(filepath, 'r') as infile:
            return infile.read()

    def _set(self, key, value, **kwargs):
        filepath = os.path.join(
            self.full_base_directory,
            self._convert_key_to_filepath(key)
        )
        path, filename = os.path.split(filepath)

        safe_mmkdir(str(path))
        with open(filepath, "wb") as outfile:
            outfile.write(value.encode("utf-8"))
        return filepath

    def list_keys(self):
        key_list = []
        for root, dirs, files in os.walk(self.full_base_directory):
            for file_ in files:
                full_path, file_name = os.path.split(os.path.join(root, file_))
                relative_path = os.path.relpath(
                    full_path,
                    self.full_base_directory,
                )
                if relative_path == ".":
                    filepath = file_name
                else:
                    filepath = os.path.join(
                        relative_path,
                        file_name
                    )

                key = self._convert_filepath_to_key(filepath)
                if key:
                    key_list.append(key)

        return key_list

    def has_key(self, key):
        assert isinstance(key, string_types)

        all_keys = self.list_keys()
        return key in all_keys


class FixedLengthTupleS3StoreBackend(FixedLengthTupleStoreBackend):
    """
    Uses an S3 bucket as a store.

    The key to this StoreBackend must be a tuple with fixed length equal to key_length.
    The filepath_template is a string template used to convert the key to a filepath.
    There's a bit of regex magic in _convert_filepath_to_key that reverses this process,
    so that we can write AND read using filenames as keys.
    """
    def __init__(
        self,
        root_directory,
        filepath_template,
        key_length,
        bucket,
        prefix="",
        boto3_options=None,
        forbidden_substrings=None,
        platform_specific_separator=False
    ):
        super(FixedLengthTupleS3StoreBackend, self).__init__(
            root_directory=root_directory,
            filepath_template=filepath_template,
            key_length=key_length,
            forbidden_substrings=forbidden_substrings,
            platform_specific_separator=platform_specific_separator
        )
        self.bucket = bucket
        self.prefix = prefix
        if boto3_options is None:
            boto3_options = {}
        self._boto3_options = boto3_options

    def _get(self, key):
        s3_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        import boto3
        s3 = boto3.client('s3', **self._boto3_options)
        s3_response_object = s3.get_object(Bucket=self.bucket, Key=s3_object_key)
        return s3_response_object['Body'].read().decode(s3_response_object.get("ContentEncoding", 'utf-8'))

    def _set(self, key, value, content_encoding='utf-8', content_type='application/json'):
        s3_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        import boto3
        s3 = boto3.resource('s3', **self._boto3_options)
        result_s3 = s3.Object(self.bucket, s3_object_key)
        result_s3.put(Body=value.encode(content_encoding), ContentEncoding=content_encoding, ContentType=content_type)
        return s3_object_key

    def list_keys(self):
        key_list = []

        import boto3
        s3 = boto3.client('s3', **self._boto3_options)

        for s3_object_info in s3.list_objects(Bucket=self.bucket, Prefix=self.prefix)['Contents']:
            s3_object_key = s3_object_info['Key']
            s3_object_key = os.path.relpath(
                s3_object_key,
                self.prefix,
            )

            key = self._convert_filepath_to_key(s3_object_key)
            if key:
                key_list.append(key)

        return key_list

    def has_key(self, key):
        assert isinstance(key, string_types)

        all_keys = self.list_keys()
        return key in all_keys


class FixedLengthTupleGCSStoreBackend(FixedLengthTupleStoreBackend):
    """
    Uses a GCS bucket as a store.

    The key to this StoreBackend must be a tuple with fixed length equal to key_length.
    The filepath_template is a string template used to convert the key to a filepath.
    There's a bit of regex magic in _convert_filepath_to_key that reverses this process,
    so that we can write AND read using filenames as keys.
    """
    def __init__(
        self,
        root_directory,
        filepath_template,
        key_length,
        bucket,
        prefix,
        project,
        forbidden_substrings=None,
        platform_specific_separator=False
    ):
        super(FixedLengthTupleGCSStoreBackend, self).__init__(
            root_directory=root_directory,
            filepath_template=filepath_template,
            key_length=key_length,
            forbidden_substrings=forbidden_substrings,
            platform_specific_separator=platform_specific_separator
        )
        self.bucket = bucket
        self.prefix = prefix
        self.project = project


    def _get(self, key):
        gcs_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        from google.cloud import storage
        gcs = storage.Client(project=self.project)
        bucket = gcs.get_bucket(self.bucket)
        gcs_response_object = bucket.get_blob(gcs_object_key)
        return gcs_response_object.download_as_string().decode("utf-8")

    def _set(self, key, value, content_encoding='utf-8', content_type='application/json'):
        gcs_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        from google.cloud import storage
        gcs = storage.Client(project=self.project)
        bucket = gcs.get_bucket(self.bucket)
        blob = bucket.blob(gcs_object_key)
        blob.upload_from_string(value.encode(content_encoding), content_type=content_type)
        return gcs_object_key

    def list_keys(self):
        key_list = []

        from google.cloud import storage
        gcs = storage.Client(self.project)

        for blob in gcs.list_blobs(self.bucket, prefix=self.prefix):
            gcs_object_name = blob.name
            gcs_object_key = os.path.relpath(
                gcs_object_name,
                self.prefix,
            )

            key = self._convert_filepath_to_key(gcs_object_key)
            if key:
                key_list.append(key)

        return key_list

    def has_key(self, key):
        assert isinstance(key, string_types)

        all_keys = self.list_keys()
        return key in all_keys
