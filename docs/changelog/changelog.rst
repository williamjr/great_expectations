.. _changelog:

0.8.2__develop
-----------------
* Use higher precision for rendering 'mostly' parameter in data-docs
* Documentation fixes (thanks @DanielOliver!)
* Minor CLI wording fixes


0.8.1
-----------------
* Fix an issue where version was reported as '0+unknown'


0.8.0
-----------------

Version 0.8.0 is a significant update to Great Expectations, with many improvements focused on configurability
and usability.  See the :ref:`migrating_versions` guide for more details on specific changes, which include
several breaking changes to configs and APIs.

Highlights include:

1. Validation Operators and Actions. Validation operators make it easy to integrate GE into a variety of pipeline runners. They
   offer one-line integration that emphasizes configurability. See the :ref:`validation_operators_and_actions`
   feature guide for more information.

   - The DataContext `get_batch` method no longer treats `expectation_suite_name` or `batch_kwargs` as optional; they
     must be explicitly specified.
   - The top-level GE validate method allows more options for specifying the specific data_asset class to use.

2. First-class support for plugins in a DataContext, with several features that make it easier to configure and
   maintain DataContexts across common deployment patterns.

   - **Environments**: A DataContext can now manage :ref:`environment_and_secrets` more easily thanks to more dynamic and
     flexible variable substitution.
   - **Stores**: A new internal abstraction for DataContexts, :ref:`stores_reference`, make extending GE easier by
     consolidating logic for reading and writing resources from a database, local, or cloud storage.
   - **Types**: Utilities configured in a DataContext are now referenced using `class_name` and `module_name` throughout
     the DataContext configuration, making it easier to extend or supplement pre-built resources. For now, the "type"
     parameter is still supported but expect it to be removed in a future release.

3. Partitioners: Batch Kwargs are clarified and enhanced to help easily reference well-known chunks of data using a
   partition_id. Batch ID and Batch Fingerprint help round out support for enhanced metadata around data
   assets that GE validates. See :ref:`batch_identifiers` for more information. The `GlobReaderGenerator`,
   `QueryGenerator`, `S3Generator`, `SubdirReaderGenerator`, and `TableGenerator` all support partition_id for
   easily accessing data assets.

4. Other Improvements:

   - We're beginning a long process of some under-the-covers refactors designed to make GE more maintainable as we
     begin adding additional features.
   - Restructured documentation: our docs have a new structure and have been reorganized to provide space for more
     easily adding and accessing reference material. Stay tuned for additional detail.
   - The command build-documentation has been renamed build-docs and now by
     default opens the Data Docs in the users' browser.

v0.7.11
-----------------
* Fix an issue where head() lost the column name for SqlAlchemyDataset objects with a single column
* Fix logic for the 'auto' bin selection of `build_continuous_partition_object`
* Add missing jinja2 dependency
* Fix an issue with inconsistent availability of strict_min and strict_max options on expect_column_values_to_be_between
* Fix an issue where expectation suite evaluation_parameters could be overriden by values during validate operation


v0.7.10
-----------------
* Fix an issue in generated documentation where the Home button failed to return to the index
* Add S3 Generator to module docs and improve module docs formatting
* Add support for views to QueryGenerator
* Add success/failure icons to index page
* Return to uniform histogram creation during profiling to avoid large partitions for internal performance reasons


v0.7.9
-----------------
* Add an S3 generator, which will introspect a configured bucket and generate batch_kwargs from identified objects
* Add support to PandasDatasource and SparkDFDatasource for reading directly from S3
* Enhance the Site Index page in documentation so that validation results are sorted and display the newest items first
  when using the default run-id scheme
* Add a new utility method, `build_continuous_partition_object` which will build partition objects using the dataset
  API and so supports any GE backend.
* Fix an issue where columns with spaces in their names caused failures in some SqlAlchemyDataset and SparkDFDataset
  expectations
* Fix an issue where generated queries including null checks failed on MSSQL (#695)
* Fix an issue where evaluation parameters passed in as a set instead of a list could cause JSON serialization problems
  for the result object (#699)


v0.7.8
-----------------
* BREAKING: slack webhook URL now must be in the profiles.yml file (treat as a secret)
* Profiler improvements:
  - Display candidate profiling data assets in alphabetical order
  - Add columns to the expectation_suite meta during profiling to support human-readable description information
* Improve handling of optional dependencies during CLI init
* Improve documentation for create_expectations notebook
* Fix several anachronistic documentation and docstring phrases (#659, #660, #668, #681; #thanks @StevenMMortimer)
* Fix data docs rendering issues:
  - documentation rendering failure from unrecognized profiled column type (#679; thanks @dinedal))
  - PY2 failure on encountering unicode (#676)


v.0.7.7
-----------------
* Standardize the way that plugin module loading works. DataContext will begin to use the new-style class and plugin
  identification moving forward; yml configs should specify class_name and module_name (with module_name optional for
  GE types). For now, it is possible to use the "type" parameter in configuration (as before).
* Add support for custom data_asset_type to all datasources
* Add support for strict_min and strict_max to inequality-based expectations to allow strict inequality checks
  (thanks @RoyalTS!)
* Add support for reader_method = "delta" to SparkDFDatasource
* Fix databricks generator (thanks @sspitz3!)
* Improve performance of DataContext loading by moving optional import
* Fix several memory and performance issues in SparkDFDataset.
  - Use only distinct value count instead of bringing values to driver
  - Migrate away from UDF for set membership, nullity, and regex expectations
* Fix several UI issues in the data_documentation
  - Move prescriptive dataset expectations to Overview section
  - Fix broken link on Home breadcrumb
  - Scroll follows navigation properly
  - Improved flow for long items in value_set
  - Improved testing for ValidationRenderer
  - Clarify dependencies introduced in documentation sites
  - Improve testing and documentation for site_builder, including run_id filter
  - Fix missing header in Index page and cut-off tooltip
  - Add run_id to path for validation files


v.0.7.6
-----------------
* New Validation Renderer! Supports turning validation results into HTML and displays differences between the expected
  and the observed attributes of a dataset.
* Data Documentation sites are now fully configurable; a data context can be configured to generate multiple
  sites built with different GE objects to support a variety of data documentation use cases. See data documentation
  guide for more detail.
* CLI now has a new top-level command, `build-documentation` that can support rendering documentation for specified
  sites and even named data assets in a specific site.
* Introduced DotDict and LooselyTypedDotDict classes that allow to enforce typing of dictionaries.
* Bug fixes: improved internal logic of rendering data documentation, slack notification, and CLI profile command when
  datasource argument was not provided.

v.0.7.5
-----------------
* Fix missing requirement for pypandoc brought in from markdown support for notes rendering.

v.0.7.4
-----------------
* Fix numerous rendering bugs and formatting issues for rendering documentation.
* Add support for pandas extension dtypes in pandas backend of expect_column_values_to_be_of_type and
  expect_column_values_to_be_in_type_list and fix bug affecting some dtype-based checks.
* Add datetime and boolean column-type detection in BasicDatasetProfiler.
* Improve BasicDatasetProfiler performance by disabling interactive evaluation when output of expectation is not
  immediately used for determining next expectations in profile.
* Add support for rendering expectation_suite and expectation_level notes from meta in docs.
* Fix minor formatting issue in readthedocs documentation.

v.0.7.3
-----------------
* BREAKING: Harmonize expect_column_values_to_be_of_type and expect_column_values_to_be_in_type_list semantics in
  Pandas with other backends, including support for None type and type_list parameters to support profiling.
  *These type expectations now rely exclusively on native python or numpy type names.*
* Add configurable support for Custom DataAsset modules to DataContext
* Improve support for setting and inheriting custom data_asset_type names
* Add tooltips with expectations backing data elements to rendered documentation
* Allow better selective disabling of tests (thanks @RoyalITS)
* Fix documentation build errors causing missing code blocks on readthedocs
* Update the parameter naming system in DataContext to reflect data_asset_name *and* expectation_suite_name
* Change scary warning about discarding expectations to be clearer, less scary, and only in log
* Improve profiler support for boolean types, value_counts, and type detection
* Allow user to specify data_assets to profile via CLI
* Support CLI rendering of expectation_suite and EVR-based documentation

v.0.7.2
-----------------
* Improved error detection and handling in CLI "add datasource" feature
* Fixes in rendering of profiling results (descriptive renderer of validation results)
* Query Generator of SQLAlchemy datasource adds tables in non-default schemas to the data asset namespace
* Added convenience methods to display HTML renderers of sections in Jupyter notebooks
* Implemented prescriptive rendering of expectations for most expectation types

v.0.7.1
------------

* Added documentation/tutorials/videos for onboarding and new profiling and documentation features
* Added prescriptive documentation built from expectation suites
* Improved index, layout, and navigation of data context HTML documentation site
* Bug fix: non-Python files were not included in the package
* Improved the rendering logic to gracefully deal with failed expectations
* Improved the basic dataset profiler to be more resilient
* Implement expect_column_values_to_be_of_type, expect_column_values_to_be_in_type_list for SparkDFDataset
* Updated CLI with a new documentation command and improved profile and render commands
* Expectation suites and validation results within a data context are saved in a more readable form (with indentation)
* Improved compatibility between SparkDatasource and InMemoryGenerator
* Optimization for Pandas column type checking
* Optimization for Spark duplicate value expectation (thanks @orenovadia!)
* Default run_id format no longer includes ":" and specifies UTC time
* Other internal improvements and bug fixes


v.0.7.0
------------

Version 0.7 of Great Expectations is HUGE. It introduces several major new features
and a large number of improvements, including breaking API changes.

The core vocabulary of expectations remains consistent. Upgrading to 
the new version of GE will primarily require changes to code that
uses data contexts; existing expectation suites will require only changes
to top-level names.

 * Major update of Data Contexts. Data Contexts now offer significantly \
   more support for building and maintaining expectation suites and \
   interacting with existing pipeline systems, including providing a namespace for objects.\
   They can handle integrating, registering, and storing validation results, and
   provide a namespace for data assets, making **batches** first-class citizens in GE.
   Read more: :ref:`data_context` or :py:mod:`great_expectations.data_context`

 * Major refactor of autoinspect. Autoinspect is now built around a module
   called "profile" which provides a class-based structure for building
   expectation suites. There is no longer a default  "autoinspect_func" --
   calling autoinspect requires explicitly passing the desired profiler. See :ref:`profiling`

 * New "Compile to Docs" feature produces beautiful documentation from expectations and expectation
   validation reports, helping keep teams on the same page.

 * Name clarifications: we've stopped using the overloaded terms "expectations
   config" and "config" and instead use "expectation suite" to refer to a
   collection (or suite!) of expectations that can be used for validating a
   data asset.

   - Expectation Suites include several top level keys that are useful \
     for organizing content in a data context: data_asset_name, \
     expectation_suite_name, and data_asset_type. When a data_asset is \
     validated, those keys will be placed in the `meta` key of the \
     validation result.

 * Major enhancement to the CLI tool including `init`, `render` and more flexibility with `validate`

 * Added helper notebooks to make it easy to get started. Each notebook acts as a combination of \
   tutorial and code scaffolding, to help you quickly learn best practices by applying them to \
   your own data.

 * Relaxed constraints on expectation parameter values, making it possible to declare many column
   aggregate expectations in a way that is always "vacuously" true, such as
   ``expect_column_values_to_be_between`` ``None`` and ``None``. This makes it possible to progressively
   tighten expectations while using them as the basis for profiling results and documentation.

  * Enabled caching on dataset objects by default.

 * Bugfixes and improvements:

   * New expectations:

     * expect_column_quantile_values_to_be_between
     * expect_column_distinct_values_to_be_in_set

   * Added support for ``head`` method on all current backends, returning a PandasDataset
   * More implemented expectations for SparkDF Dataset with optimizations

     * expect_column_values_to_be_between
     * expect_column_median_to_be_between
     * expect_column_value_lengths_to_be_between

   * Optimized histogram fetching for SqlalchemyDataset and SparkDFDataset
   * Added cross-platform internal partition method, paving path for improved profiling
   * Fixed bug with outputstrftime not being honored in PandasDataset
   * Fixed series naming for column value counts
   * Standardized naming for expect_column_values_to_be_of_type
   * Standardized and made explicit use of sample normalization in stdev calculation
   * Added from_dataset helper
   * Internal testing improvements
   * Documentation reorganization and improvements
   * Introduce custom exceptions for more detailed error logs

v.0.6.1
------------
* Re-add testing (and support) for py2
* NOTE: Support for SqlAlchemyDataset and SparkDFDataset is enabled via optional install \
  (e.g. ``pip install great_expectations[sqlalchemy]`` or ``pip install great_expectations[spark]``)

v.0.6.0
------------
* Add support for SparkDFDataset and caching (HUGE work from @cselig)
* Migrate distributional expectations to new testing framework
* Add support for two new expectations: expect_column_distinct_values_to_contain_set 
  and expect_column_distinct_values_to_equal_set (thanks @RoyalTS)
* FUTURE BREAKING CHANGE: The new cache mechanism for Datasets, \
  when enabled, causes GE to assume that dataset does not change between evaluation of individual expectations. \
  We anticipate this will become the future default behavior.
* BREAKING CHANGE: Drop official support pandas < 0.22

v.0.5.1
---------------
* **Fix** issue where no result_format available for expect_column_values_to_be_null caused error
* Use vectorized computation in pandas (#443, #445; thanks @RoyalTS)


v.0.5.0
----------------
* Restructured class hierarchy to have a more generic DataAsset parent that maintains expectation logic separate \
  from the tabular organization of Dataset expectations
* Added new FileDataAsset and associated expectations (#416 thanks @anhollis)
* Added support for date/datetime type columns in some SQLAlchemy expectations (#413)
* Added support for a multicolumn expectation, expect multicolumn values to be unique (#408)
* **Optimization**: You can now disable `partial_unexpected_counts` by setting the `partial_unexpected_count` value to \
  0 in the result_format argument, and we do not compute it when it would not be returned. (#431, thanks @eugmandel)
* **Fix**: Correct error in unexpected_percent computations for sqlalchemy when unexpected values exceed limit (#424)
* **Fix**: Pass meta object to expectation result (#415, thanks @jseeman)
* Add support for multicolumn expectations, with `expect_multicolumn_values_to_be_unique` as an example (#406)
* Add dataset class to from_pandas to simplify using custom datasets (#404, thanks @jtilly)
* Add schema support for sqlalchemy data context (#410, thanks @rahulj51)
* Minor documentation, warning, and testing improvements (thanks @zdog).


v.0.4.5
----------------
* Add a new autoinspect API and remove default expectations.
* Improve details for expect_table_columns_to_match_ordered_list (#379, thanks @rlshuhart)
* Linting fixes (thanks @elsander)
* Add support for dataset_class in from_pandas (thanks @jtilly)
* Improve redshift compatibility by correcting faulty isnull operator (thanks @avanderm)
* Adjust partitions to use tail_weight to improve JSON compatibility and
  support special cases of KL Divergence (thanks @anhollis)
* Enable custom_sql datasets for databases with multiple schemas, by
  adding a fallback for column reflection (#387, thanks @elsander)
* Remove `IF NOT EXISTS` check for custom sql temporary tables, for
  Redshift compatibility (#372, thanks @elsander)
* Allow users to pass args/kwargs for engine creation in
  SqlAlchemyDataContext (#369, thanks @elsander)
* Add support for custom schema in SqlAlchemyDataset (#370, thanks @elsander)
* Use getfullargspec to avoid deprecation warnings.
* Add expect_column_values_to_be_unique to SqlAlchemyDataset
* **Fix** map expectations for categorical columns (thanks @eugmandel)
* Improve internal testing suite (thanks @anhollis and @ccnobbli)
* Consistently use value_set instead of mixing value_set and values_set (thanks @njsmith8)

v.0.4.4
----------------
* Improve CLI help and set CLI return value to the number of unmet expectations
* Add error handling for empty columns to SqlAlchemyDataset, and associated tests
* **Fix** broken support for older pandas versions (#346)
* **Fix** pandas deepcopy issue (#342)

v.0.4.3
-------
* Improve type lists in expect_column_type_to_be[_in_list] (thanks @smontanaro and @ccnobbli)
* Update cli to use entry_points for conda compatibility, and add version option to cli
* Remove extraneous development dependency to airflow
* Address SQlAlchemy warnings in median computation
* Improve glossary in documentation
* Add 'statistics' section to validation report with overall validation results (thanks @sotte)
* Add support for parameterized expectations
* Improve support for custom expectations with better error messages (thanks @syk0saje)
* Implement expect_column_value_lenghts_to_[be_between|equal] for SQAlchemy (thanks @ccnobbli)
* **Fix** PandasDataset subclasses to inherit child class

v.0.4.2
-------
* **Fix** bugs in expect_column_values_to_[not]_be_null: computing unexpected value percentages and handling all-null (thanks @ccnobbli)
* Support mysql use of Decimal type (thanks @bouke-nederstigt)
* Add new expectation expect_column_values_to_not_match_regex_list.

  * Change behavior of expect_column_values_to_match_regex_list to use python re.findall in PandasDataset, relaxing \
    matching of individuals expressions to allow matches anywhere in the string.

* **Fix** documentation errors and other small errors (thanks @roblim, @ccnobbli)

v.0.4.1
-------
* Correct inclusion of new data_context module in source distribution

v.0.4.0
-------
* Initial implementation of data context API and SqlAlchemyDataset including implementations of the following \
  expectations:

  * expect_column_to_exist
  * expect_table_row_count_to_be
  * expect_table_row_count_to_be_between
  * expect_column_values_to_not_be_null
  * expect_column_values_to_be_null
  * expect_column_values_to_be_in_set
  * expect_column_values_to_be_between
  * expect_column_mean_to_be
  * expect_column_min_to_be
  * expect_column_max_to_be
  * expect_column_sum_to_be
  * expect_column_unique_value_count_to_be_between
  * expect_column_proportion_of_unique_values_to_be_between

* Major refactor of output_format to new result_format parameter. See docs for full details:

  * exception_list and related uses of the term exception have been renamed to unexpected
  * Output formats are explicitly hierarchical now, with BOOLEAN_ONLY < BASIC < SUMMARY < COMPLETE. \
    All *column_aggregate_expectation* expectations now return element count and related information included at the \
    BASIC level or higher.

* New expectation available for parameterized distributions--\
  expect_column_parameterized_distribution_ks_test_p_value_to_be_greater_than (what a name! :) -- (thanks @ccnobbli)
* ge.from_pandas() utility (thanks @schrockn)
* Pandas operations on a PandasDataset now return another PandasDataset (thanks @dlwhite5)
* expect_column_to_exist now takes a column_index parameter to specify column order (thanks @louispotok)
* Top-level validate option (ge.validate())
* ge.read_json() helper (thanks @rjurney)
* Behind-the-scenes improvements to testing framework to ensure parity across data contexts.
* Documentation improvements, bug-fixes, and internal api improvements

v.0.3.2
-------
* Include requirements file in source dist to support conda

v.0.3.1
--------
* **Fix** infinite recursion error when building custom expectations
* Catch dateutil parsing overflow errors

v.0.2
-----
* Distributional expectations and associated helpers are improved and renamed to be more clear regarding the tests they apply
* Expectation decorators have been refactored significantly to streamline implementing expectations and support custom expectations
* API and examples for custom expectations are available
* New output formats are available for all expectations
* Significant improvements to test suite and compatibility
