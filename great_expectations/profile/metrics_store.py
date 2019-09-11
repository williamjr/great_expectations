import collections
from functools import reduce
import operator
from collections import defaultdict
from great_expectations.data_context.types.metrics import (
NamespaceAwareValidationMetric,
MultiBatchNamespaceAwareValidationMetric,
NamespaceAwareExpectationDefinedValidationMetric,
MultiBatchNamespaceAwareExpectationDefinedValidationMetric,
)


class MetricsStore(object):
    EXPECTATION_DEFINED_METRICS_LOOKUP_TABLE = {
        ('expect_column_values_to_not_be_null', 'unexpected_percent'): ('column',), # note: "," is important - it makes it a tuple!
        ('expect_column_kl_divergence_to_be_less_than', ('details', 'observed_partition', 'bins',)): ('column',),
        ('expect_column_values_to_be_between', 'unexpected_percent'): ('column', 'min_value', 'max_value'),
        ('expect_column_values_to_be_in_type_list', 'unexpected_percent'): ('column', 'type_list'),
        ('expect_column_values_to_be_unique', 'unexpected_percent'): ('column',),
        ('expect_column_values_to_not_be_null', 'unexpected_percent'): ('column',),
        ('expect_column_values_to_not_match_regex', 'unexpected_percent'): ('column', 'regex')
    }

    @classmethod
    def add_expectation_defined_metric_for_result_key(cls, d, result, data_asset_name, batch_fingerprint, t=()):
        for key, value in d.items():
            if isinstance(value, collections.Mapping):
                cls.add_expectation_defined_metric_for_result_key(value, result, data_asset_name, batch_fingerprint, t + (key,))
            else:
                result_key_lookup_key = key if t==() else (t + (key,))
                full_lookup_key = (result['expectation_config']['expectation_type'], result_key_lookup_key)
                metric_kwargs_names = cls.EXPECTATION_DEFINED_METRICS_LOOKUP_TABLE.get(full_lookup_key)
                if metric_kwargs_names:
                    metric_kwargs = {}
                    for metric_kwarg_name in metric_kwargs_names:
                        if isinstance(metric_kwarg_name, tuple):
                            set_nested_value_in_dict(metric_kwargs, metric_kwarg_name, get_nested_value_from_dict(result['expectation_config']['kwargs'], metric_kwarg_name))
                        else:
                            metric_kwargs[metric_kwarg_name] = result['expectation_config']['kwargs'][metric_kwarg_name]

                    new_metric = NamespaceAwareExpectationDefinedValidationMetric(
                        data_asset_name=data_asset_name,
                        batch_fingerprint=batch_fingerprint,
                        expectation_type=result['expectation_config']['expectation_type'],
                        result_key=result_key_lookup_key,
                        metric_kwargs=metric_kwargs,
                        metric_value=value)

                    yield new_metric

    @classmethod
    def get_metrics_for_expectation(cls, result, data_asset_name, batch_fingerprint):
        """
        Return a list of multi batch metrics for a list of batches
        :param batch_fingerprints:
        :return: dict of multi batch metrics (by mb metric key).
                Values are MultiBatchNamespaceAwareValidationMetric or
                MultiBatchNamespaceAwareExpectationDefinedValidationMetric
        """
        expectation_metrics = {
            'expect_column_distinct_values_to_be_in_set': {
                'observed_value': 'distinct_set_members'
            },
            'expect_column_max_to_be_between': {
                'observed_value': 'column_max'
            },
            'expect_column_mean_to_be_between': {
                'observed_value': 'column_mean'
            },
            'expect_column_median_to_be_between': {
                'observed_value': 'column_median'
            },
            'expect_column_min_to_be_between': {
                'observed_value': 'column_min'
            },
            'expect_column_proportion_of_unique_values_to_be_between': {
                'observed_value': 'column_proportion_of_unique_values'
            },
            'expect_column_quantile_values_to_be_between': {
                'observed_value': 'column_quantiles'
            },
            'expect_column_stdev_to_be_between': {
                'observed_value': 'column_stdev'
            },
            'expect_column_unique_value_count_to_be_between': {
                'observed_value': 'column_unique_count'
            },
            # 'expect_column_values_to_be_in_set',
            # 'expect_table_columns_to_match_ordered_list',
            'expect_table_row_count_to_be_between': {
                'observed_value': 'row_count'
            }
        }

        metrics = []
        if result.get('result'):
            entry = expectation_metrics.get(result['expectation_config']['expectation_type'])
            if entry:
                for key in result['result'].keys():
                    metric_name = entry.get(key)
                    if metric_name:
                        metric_kwargs = {"column": result['expectation_config']['kwargs']['column']} if result['expectation_config'][
                    'kwargs'].get('column') else {}

                        new_metric = NamespaceAwareValidationMetric(
                            data_asset_name=data_asset_name,
                            batch_fingerprint=batch_fingerprint,
                            expectation_type=result['expectation_config']['expectation_type'],
                            metric_name=metric_name,
                            metric_kwargs=metric_kwargs,
                            metric_value=result['result'][key])
                        metrics.append(new_metric)
            else:
                for new_metric in cls.add_expectation_defined_metric_for_result_key(result['result'], result, data_asset_name, batch_fingerprint):
                    metrics.append(new_metric)

        return metrics

        dict_selected_batches = {}
        for batch_fingerprint, batch_metrics in self.dict_single_batch_metrics_by_multi_batch_key_by_batch.items():
            if batch_fingerprint in [bk.batch_fingerprint.fingerprint for bk in batch_kwargs_list]:
                dict_selected_batches[batch_fingerprint] = batch_metrics

        # let's compute the union of all metrics names that come from all the batches.
        # this will help us fill with nulls if a particular metric is missing from a batch
        # (e.g., due to the column missing)
        # Not performing this steps would result in non-uniform lengths of lists and we would
        # not be able to convert this dict of lists into a dataframe.
        metric_names_union = set()
        for batch_id, batch_metrics in dict_selected_batches.items():
            metric_names_union = metric_names_union.union(batch_metrics.keys())

        metrics_dict_of_lists = defaultdict(list)

        batch_index = list(self.dict_single_batch_metrics_by_multi_batch_key_by_batch.keys())

        for batch_id, batch_metrics in dict_selected_batches.items():
            # fill in the metrics that are present in the batch
            for metric_name, metric_value in batch_metrics.items():
                metrics_dict_of_lists[metric_name].append(metric_value)

            # fill in the metrics that are missing in the batch
            metrics_missing_in_batch = metric_names_union - set(batch_metrics.keys())
            for metric_name in metrics_missing_in_batch:
                metrics_dict_of_lists[metric_name].append(None)

        mb_metrics = {}
        for metric_key, single_batch_metric_list in metrics_dict_of_lists.items():
            mb_metric = self._make_multi_batch_metric_from_list_of_single_batch_metrics(metric_key[0], single_batch_metric_list,
                                                                                      batch_index)
            mb_metrics[mb_metric.key] = mb_metric

        return mb_metrics

    def _make_multi_batch_metric_from_list_of_single_batch_metrics(self, single_batch_metric_name, single_batch_metric_list, batch_index):
        """
        Utility method that gets a list of single batch metrics with the same multi-batch key (meaning that they are the same
        metric with the same kwargs, but obtained by validating different batches of the same data asset) and
        constructs a multi-batch metric for that key.

        :param single_batch_metric_name:
        :param single_batch_metric_list:
        :param batch_index:
        :return:
        """
        first_non_null_single_batch_metric = [item for item in single_batch_metric_list if item is not None][0]

        if 'NamespaceAwareValidationMetric' == single_batch_metric_name:
                mb_metric = MultiBatchNamespaceAwareValidationMetric(
                    data_asset_name=first_non_null_single_batch_metric.data_asset_name,
                    metric_name=first_non_null_single_batch_metric.metric_name,
                    metric_kwargs=first_non_null_single_batch_metric.metric_kwargs,
                    batch_fingerprints=batch_index,
                    batch_metric_values=[None if metric is None else metric.metric_value for metric in
                                         single_batch_metric_list]
                )
        elif 'NamespaceAwareExpectationDefinedValidationMetric' == single_batch_metric_name:
                mb_metric = MultiBatchNamespaceAwareExpectationDefinedValidationMetric(
                    data_asset_name = first_non_null_single_batch_metric.data_asset_name,
                    result_key = first_non_null_single_batch_metric.result_key,
                    expectation_type = first_non_null_single_batch_metric.expectation_type,
                    metric_kwargs = first_non_null_single_batch_metric.metric_kwargs,
                    batch_fingerprints = batch_index,
                    batch_metric_values = [None if metric is None else metric.metric_value for metric in single_batch_metric_list]
                )

        return mb_metric
