.. _distributional_expectations:

================================================================================
Distributional Expectations
================================================================================

Distributional expectations help identify when new datasets or samples may be different than expected, and can help \
ensure that assumptions developed during exploratory analysis still hold as new data becomes available. You should use \
distributional expectations in the same way as other expectations: to help accelerate identification of risks as \
diverse as changes in a system being modeled or disruptions to a complex upstream data feed.

Great Expectations' Philosophy of Distributional Expectations
--------------------------------------------------------------------------------

Great Expectations attempts to provide a simple, expressive framework for describing distributional expectations. \
The framework generally adopts a nonparametric approach, although it is possible to build expectations from \
parameterized distributions.

The design is motivated by the following assumptions:

* Encoding expectations into a simple object that allows for portable data pipeline testing is the top priority. \
  In many circumstances the loss of precision associated with "compressing" data into an expectation may be beneficial \
  because of its intentional simplicity as well as because it adds a very light layer of obfuscation over the data \
  which may align with privacy preservation goals.
* While it should be possible to easily extend the framework with more rigorous statistical tests, great expectations \
  should provide simple, reasonable defaults. Care should be taken in cases where robust statistical guarantees are \
  expected.
* Building and interpreting expectations should be intuitive: a more precise partition object implies a more precise \
  expectation.


.. _partition_object:

Partition Objects
--------------------------------------------------------------------------------

The core constructs of a great expectations distributional expectation are the partition and associated weights.

For continuous data:

* A partition is defined by an ordered list of points that define intervals on the real number line. Note that partition intervals do not need to be uniform.
* Each bin in a partition is partially open: a data element x is in bin i if lower_bound_i <= x < upper_bound_i.
* However, following the behavior of numpy.histogram, a data element x is in the largest bin k if x == upper_bound_k.
* A bin may include -Infinity and Infinity as endpoints, however, those endpoints are not supported by the Kolmogorov-Smirnov test.

* Partition weights define the probability of the associated interval. Note that this effectively applies a "piecewise uniform" distribution to the data for the purpose of statistical tests. The weights must define a valid probability distribution, ie they must be non-negative numbers that sum to 1.

Example continuous partition object:

.. code-block:: python

  {
    "bins": [ 0, 1, 2, 10],
    "weights": [0.3, 0.3, 0.4]
  }

For discrete/categorical data:

* A partition defines the categorical values present in the data.
* Partition weights define the probability of the associated categorical value.

Example discrete partition object:

.. code-block:: python

  {
    "values": [ "cat", "dog", "fish"],
    "weights": [0.3, 0.3, 0.4]
  }


Constructing Partition Objects
--------------------------------------------------------------------------------
Three convenience functions are available to easily construct partition objects from existing data:

* :func:`continuous_partition_data <great_expectations.dataset.util.continuous_partition_data>`
* :func:`categorical_partition_data <great_expectations.dataset.util.categorical_partition_data>`
* :func:`kde_partition_data <great_expectations.dataset.util.kde_partition_data>`

Convenience functions are also provided to validate that an object is a valid partition density object:

* :func:`is_valid_continuous_partition_object <great_expectations.dataset.util.is_valid_continuous_partition_object>`
* :func:`is_valid_categorical_partition_object <great_expectations.dataset.util.is_valid_categorical_partition_object>`

Tests interpret partition objects literally, so care should be taken when a partition includes a segment with zero weight. The convenience methods consequently allow you to include small amounts of residual weight on the "tails" of a dataset used to construct a partition.


Distributional Expectations Core Tests
--------------------------------------------------------------------------------
Distributional expectations rely on three tests for their work.

Kullback-Leibler (KL) divergence is available as an expectation for both categorical and continuous data (continuous data will be discretized according to the provided partition prior to computing divergence). Unlike KS and Chi-Squared tests which can use a p-value, you must provide a threshold for the relative entropy to use KL divergence. Further, KL divergence is not symmetric.

* :func:`expect_column_kl_divergence_to_be_less_than <great_expectations.dataset.dataset.Dataset.expect_column_kl_divergence_to_be_less_than>`

For continuous data, the expect_column_bootstrapped_ks_test_p_value_to_be_greater_than expectation uses the Kolmogorov-Smirnov (KS) test, which compares the actual and expected cumulative densities of the data. Because of the partition_object's piecewise uniform approximation of the expected distribution, the test would be overly sensitive to differences when used with a sample of data of much larger than the size of the partition. The expectation consequently uses a bootstrapping method to sample the provided data with tunable specificity.

* :func:`expect_column_bootstrapped_ks_test_p_value_to_be_greater_than <great_expectations.dataset.dataset.Dataset.expect_column_bootstrapped_ks_test_p_value_to_be_greater_than>`

For categorical data, the expect_column_chisquare_test_p_value_to_be_greater_than expectation uses the Chi-Squared test. The Chi-Squared test works with expected and observed counts, but that is handled internally in this function -- both the input and output to this function are valid partition objects (ie with weights that are probabilities and sum to 1).

* :func:`expect_column_chisquare_test_p_value_to_be_greater_than <great_expectations.dataset.dataset.Dataset.expect_column_chisquare_test_p_value_to_be_greater_than>`



Distributional Expectations Alternatives
--------------------------------------------------------------------------------
The core partition density object used in current expectations focuses on a particular (partition-based) method of "compressing" the data into a testable form, however it may be desireable to use alternative nonparametric approaches (e.g. Fourier transform/wavelets) to describe expected data.
