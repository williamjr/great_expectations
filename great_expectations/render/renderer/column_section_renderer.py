from functools import reduce
import json
from string import Template
import inspect
import re

import altair as alt
import pandas as pd

from .renderer import Renderer
from .content_block import ValueListContentBlockRenderer
from .content_block import TableContentBlockRenderer
from .content_block import (ExpectationSuiteBulletListContentBlockRenderer)
from great_expectations.render.renderer.content_block import ValidationResultsTableContentBlockRenderer
from .content_block import ExceptionListContentBlockRenderer

from ..types import RenderedSectionContent

from ..types import (
    RenderedComponentContent,
)

def convert_to_string_and_escape(var):
    return re.sub("\$", "$$", str(var))

class ColumnSectionRenderer(Renderer):
    @classmethod
    def _get_column_name(cls, ge_object):
        # This is broken out for ease of locating future validation here
        if isinstance(ge_object, list):
            candidate_object = ge_object[0]
        else:
            candidate_object = ge_object
        try:
            if "kwargs" in candidate_object:
                # This is an expectation
                return candidate_object["kwargs"]["column"]
            elif "expectation_config" in candidate_object:
                # This is a validation
                return candidate_object["expectation_config"]["kwargs"]["column"]
            else:
                raise ValueError(
                    "Provide a column section renderer an expectation, list of expectations, evr, or list of evrs.")
        except KeyError:
            return None


class ProfilingResultsColumnSectionRenderer(ColumnSectionRenderer):

    #Note: Seems awkward to pass section_name and column_type into this renderer.
    #Can't we figure that out internally?
    @classmethod
    def render(cls, evrs, section_name=None, column_type=None):
        if section_name is None:
            column = cls._get_column_name(evrs)
        else:
            column = section_name

        content_blocks = []

        content_blocks.append(cls._render_header(evrs, column_type))
        # content_blocks.append(cls._render_column_type(evrs))
        content_blocks.append(cls._render_overview_table(evrs))
        content_blocks.append(cls._render_quantile_table(evrs))
        content_blocks.append(cls._render_stats_table(evrs))
        content_blocks.append(cls._render_histogram(evrs))
        content_blocks.append(cls._render_values_set(evrs))
        content_blocks.append(cls._render_bar_chart_table(evrs))

        # content_blocks.append(cls._render_statistics(evrs))
        # content_blocks.append(cls._render_common_values(evrs))
        # content_blocks.append(cls._render_extreme_values(evrs))
        # content_blocks.append(cls._render_frequency(evrs))
        # content_blocks.append(cls._render_composition(evrs))
        # content_blocks.append(cls._render_expectation_types(evrs))
        # content_blocks.append(cls._render_unrecognized(evrs))

        content_blocks.append(cls._render_failed(evrs))
        # NOTE : Some render* functions return None so we filter them out
        populated_content_blocks = list(filter(None, content_blocks))

        return RenderedSectionContent(**{
            "section_name": column,
            "content_blocks": populated_content_blocks,
        })

    @classmethod
    def _render_header(cls, evrs, column_type=None):
        # NOTE: This logic is brittle
        try:
            column_name = evrs[0]["expectation_config"]["kwargs"]["column"]
        except KeyError:
            column_name = "Table-level expectations"

        return RenderedComponentContent(**{
            "content_block_type": "header",
            "header": {
                    "template": convert_to_string_and_escape(column_name),
                    "tooltip": {
                        "content": "expect_column_to_exist",
                        "placement": "top"
                    },
                },
            "subheader": {
                    "template": "Type: {column_type}".format(column_type=column_type),
                    "tooltip": {
                      "content": "expect_column_values_to_be_of_type <br>expect_column_values_to_be_in_type_list",
                    },
                },
            # {
            #     "template": column_type,
            # },
            "styling": {
                "classes": ["col-12"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })

    @classmethod
    def _render_expectation_types(cls, evrs, content_blocks):
        # NOTE: The evr-fetching function is an kinda similar to the code other_section_
        # renderer.ProfilingResultsOverviewSectionRenderer._render_expectation_types

        # type_counts = defaultdict(int)

        # for evr in evrs:
        #     type_counts[evr["expectation_config"]["expectation_type"]] += 1

        # bullet_list = sorted(type_counts.items(), key=lambda kv: -1*kv[1])

        bullet_list = [{
            "content_block_type": "string_template",
            "string_template": {
                "template": "$expectation_type $is_passing",
                "params": {
                    "expectation_type": evr["expectation_config"]["expectation_type"],
                    "is_passing": str(evr["success"]),
                },
                "styling": {
                    "classes": ["list-group-item", "d-flex", "justify-content-between", "align-items-center"],
                    "params": {
                        "is_passing": {
                            "classes": ["badge", "badge-secondary", "badge-pill"],
                        }
                    },
                }
            }
        } for evr in evrs]

        content_blocks.append(RenderedComponentContent(**{
            "content_block_type": "bullet_list",
            "header": 'Expectation types <span class="mr-3 triangle"></span>',
            "bullet_list": bullet_list,
            "styling": {
                "classes": ["col-12"],
                "styles": {
                    "margin-top": "20px"
                },
                "header": {
                    # "classes": ["alert", "alert-secondary"],
                    "classes": ["collapsed"],
                    "attributes": {
                        "data-toggle": "collapse",
                        "href": "#{{content_block_id}}-body",
                        "role": "button",
                        "aria-expanded": "true",
                        "aria-controls": "collapseExample",
                    },
                    "styles": {
                        "cursor": "pointer",
                    }
                },
                "body": {
                    "classes": ["list-group", "collapse"],
                },
            },
        }))

    @classmethod
    def _render_overview_table(cls, evrs):
        unique_n = cls._find_evr_by_type(
            evrs,
            "expect_column_unique_value_count_to_be_between"
        )
        unique_proportion = cls._find_evr_by_type(
            evrs,
            "expect_column_proportion_of_unique_values_to_be_between"
        )
        null_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_values_to_not_be_null"
        )
        evrs = [evr for evr in [unique_n, unique_proportion, null_evr] if (evr is not None and "result" in evr)]

        if len(evrs) > 0:
            new_content_block = TableContentBlockRenderer.render(evrs)
            new_content_block["header"] = "Properties"
            new_content_block["styling"] = {
                "classes": ["col-4", ],
                "styles": {
                    "margin-top": "20px"
                },
                "body": {
                    "classes": ["table", "table-sm", "table-unbordered"],
                    "styles": {
                        "width": "100%"
                    },
                }

            }
            return new_content_block

    @classmethod
    def _render_quantile_table(cls, evrs):
        table_rows = []

        quantile_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_quantile_values_to_be_between"
        )

        if not quantile_evr or "result" not in quantile_evr:
            return

        quantiles = quantile_evr["result"]["observed_value"]["quantiles"]
        quantile_ranges = quantile_evr["result"]["observed_value"]["values"]

        quantile_strings = {
            .25: "Q1",
            .75: "Q3",
            .50: "Median"
        }
        
        for idx, quantile in enumerate(quantiles):
            quantile_string = quantile_strings.get(quantile)
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": quantile_string if quantile_string else "{:3.2f}".format(quantile),
                        "tooltip": {
                            "content": "expect_column_quantile_values_to_be_between \n expect_column_median_to_be_between" if quantile == 0.50 else "expect_column_quantile_values_to_be_between"
                        }
                    }
                },
                quantile_ranges[idx],
            ])

        return RenderedComponentContent(**{
            "content_block_type": "table",
            "header": "Quantiles",
            "table": table_rows,
            "styling": {
                "classes": ["col-4"],
                "styles": {
                    "margin-top": "20px"
                },
                "body": {
                    "classes": ["table", "table-sm", "table-unbordered"],
                }
            },
        })

    @classmethod
    def _render_stats_table(cls, evrs):
        table_rows = []

        mean_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_mean_to_be_between"
        )

        if not mean_evr or "result" not in mean_evr:
            return

        mean_value = "{:.2f}".format(
            mean_evr['result']['observed_value']) if mean_evr else None
        if mean_value:
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Mean",
                        "tooltip": {
                            "content": "expect_column_mean_to_be_between"
                        }
                    }
                },
                mean_value
            ])

        min_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_min_to_be_between"
        )
        min_value = "{:.2f}".format(
            min_evr['result']['observed_value']) if min_evr else None
        if min_value:
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Minimum",
                        "tooltip": {
                            "content": "expect_column_min_to_be_between"
                        }
                    }
                },
                min_value,
            ])

        max_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_max_to_be_between"
        )
        max_value = "{:.2f}".format(
            max_evr['result']['observed_value']) if max_evr else None
        if max_value:
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Maximum",
                        "tooltip": {
                            "content": "expect_column_max_to_be_between"
                        }
                    }
                },
                max_value
            ])

        if len(table_rows) > 0:
            return RenderedComponentContent(**{
                "content_block_type": "table",
                "header": "Statistics",
                "table": table_rows,
                "styling": {
                    "classes": ["col-4"],
                    "styles": {
                        "margin-top": "20px"
                    },
                    "body": {
                        "classes": ["table", "table-sm", "table-unbordered"],
                    }
                },
            })
        else:
            return

    @classmethod
    def _render_values_set(cls, evrs):
        set_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_values_to_be_in_set"
        )

        if not set_evr or "result" not in set_evr:
            return

        if set_evr and "partial_unexpected_counts" in set_evr["result"]:
            partial_unexpected_counts = set_evr["result"]["partial_unexpected_counts"]
            values = [str(v["value"]) for v in partial_unexpected_counts]
        elif set_evr and "partial_unexpected_list" in set_evr["result"]:
            values = [str(item) for item in set_evr["result"]["partial_unexpected_list"]]
        else:
            return

        if len(" ".join(values)) > 100:
            classes = ["col-12"]
        else:
            classes = ["col-4"]

        if any(len(value) > 80 for value in values):
            content_block_type = "bullet_list"
        else:
            content_block_type = "value_list"

        new_block = RenderedComponentContent(**{
            "content_block_type": content_block_type,
            "header":
                {
                    "template": "Example Values",
                    "tooltip": {
                        "content": "expect_column_values_to_be_in_set"
                    }
                },
            content_block_type: [{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "$value",
                    "params": {
                        "value": value
                    },
                    "styling": {
                        "default": {
                            "classes": ["badge", "badge-info"] if content_block_type == "value_list" else [],
                            "styles": {
                                "word-break": "break-all"
                            }
                        },
                    }
                }
            } for value in values],
            "styling": {
                "classes": classes,
                "styles": {
                    "margin-top": "20px",
                }
            }
        })

        return new_block


    @classmethod
    def _render_histogram(cls, evrs):
        # NOTE: This code is very brittle
        kl_divergence_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_kl_divergence_to_be_less_than"
        )
        # print(json.dumps(kl_divergence_evr, indent=2))
        if not kl_divergence_evr or "result" not in kl_divergence_evr or "details" not in kl_divergence_evr.get("result", {}):
            return

        bins = kl_divergence_evr["result"]["details"]["observed_partition"]["bins"]
        # bin_medians = [round((v+bins[i+1])/2, 1)
        #                for i, v in enumerate(bins[:-1])]
        # bin_medians = [(round(bins[i], 1), round(bins[i+1], 1)) for i, v in enumerate(bins[:-1])]
        bins_x1 = [round(value, 1) for value in bins[:-1]]
        bins_x2 = [round(value, 1) for value in bins[1:]]
        weights = kl_divergence_evr["result"]["details"]["observed_partition"]["weights"]

        df = pd.DataFrame({
            "bin_min": bins_x1,
            "bin_max": bins_x2,
            "weights": weights,
        })
        df.weights *= 100

        if len(weights) <= 10:
            height = 200
            width = 200
            col_width = 4
        else:
            height = 300
            width = 300
            col_width = 6

        bars = alt.Chart(df).mark_bar().encode(
            x='bin_min:O',
            x2='bin_max:O',
            y="weights:Q"
        ).properties(width=width, height=height, autosize="fit")

        chart = bars.to_json()

        return RenderedComponentContent(**{
            "content_block_type": "graph",
            "header":
                {
                    "template": "Histogram",
                    "tooltip": {
                        "content": "expect_column_kl_divergence_to_be_less_than"
                    }
                },
            "graph": chart,
            "styling": {
                "classes": ["col-" + str(col_width)],
                "styles": {
                    "margin-top": "20px",
                }
            }
        })


    @classmethod
    def _render_bar_chart_table(cls, evrs):
        distinct_values_set_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_distinct_values_to_be_in_set"
        )
        # print(json.dumps(kl_divergence_evr, indent=2))
        if not distinct_values_set_evr or "result" not in distinct_values_set_evr:
            return

        value_count_dicts = distinct_values_set_evr['result']['details']['value_counts']
        values = [value_count_dict['value']
                  for value_count_dict in value_count_dicts]
        counts = [value_count_dict['count']
                  for value_count_dict in value_count_dicts]

        df = pd.DataFrame({
            "value": values,
            "count": counts,
        })

        if len(values) <= 10:
            height = 200
            width = 200
            col_width = 4
        else:
            height = 300
            width = 300
            col_width = 6

        bars = alt.Chart(df).mark_bar(size=20).encode(
            y='count:Q',
            x="value:O"
        ).properties(height=height, width=width, autosize="fit")

        chart = bars.to_json()

        new_block = RenderedComponentContent(**{
            "content_block_type": "graph",
            "header":
                {
                    "template": "Value Counts",
                    "tooltip": {
                        "content": "expect_column_distinct_values_to_be_in_set"
                    }
                },
            "graph": chart,
            "styling": {
                "classes": ["col-" + str(col_width)],
                "styles": {
                    "margin-top": "20px",
                }
            }
        })

        return new_block

    @classmethod
    def _render_failed(cls, evrs):
        return ExceptionListContentBlockRenderer.render(evrs, include_column_name=False)

    @classmethod
    def _render_unrecognized(cls, evrs, content_blocks):
        unrendered_blocks = []
        new_block = None
        for evr in evrs:
            if evr["expectation_config"]["expectation_type"] not in [
                "expect_column_to_exist",
                "expect_column_values_to_be_of_type",
                "expect_column_values_to_be_in_set",
                "expect_column_unique_value_count_to_be_between",
                "expect_column_proportion_of_unique_values_to_be_between",
                "expect_column_values_to_not_be_null",
                "expect_column_max_to_be_between",
                "expect_column_mean_to_be_between",
                "expect_column_min_to_be_between"
            ]:
                new_block = RenderedComponentContent(**{
                    "content_block_type": "text",
                    "content": []
                })
                new_block["content"].append("""
    <div class="alert alert-primary" role="alert">
        Warning! Unrendered EVR:<br/>
    <pre>"""+json.dumps(evr, indent=2)+"""</pre>
    </div>
                """)

        if new_block is not None:
            unrendered_blocks.append(new_block)

        # print(unrendered_blocks)
        content_blocks += unrendered_blocks


class ValidationResultsColumnSectionRenderer(ColumnSectionRenderer):
    @classmethod
    def _render_header(cls, validation_results):
        column = cls._get_column_name(validation_results)
        
        new_block = RenderedComponentContent(**{
            "content_block_type": "header",
            "header": convert_to_string_and_escape(column),
            "styling": {
                "classes": ["col-12"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })
        
        return validation_results, new_block
    
    @classmethod
    def _render_table(cls, validation_results):
        new_block = ValidationResultsTableContentBlockRenderer.render(
            validation_results,
            include_column_name=False
        )
        
        return [], new_block
    
    @classmethod
    def render(cls, validation_results={}):
        column = cls._get_column_name(validation_results)
        content_blocks = []
        remaining_evrs, content_block = cls._render_header(validation_results)
        content_blocks.append(content_block)
        remaining_evrs, content_block = cls._render_table(remaining_evrs)
        content_blocks.append(content_block)

        return RenderedSectionContent(**{
            "section_name": column,
            "content_blocks": content_blocks
        })


class ExpectationSuiteColumnSectionRenderer(ColumnSectionRenderer):

    @classmethod
    def _render_header(cls, expectations):
        column = cls._get_column_name(expectations)

        new_block = RenderedComponentContent(**{
            "content_block_type": "header",
            "header": convert_to_string_and_escape(column),
            "styling": {
                "classes": ["col-12"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })

        return expectations, new_block

    @classmethod
    def _render_bullet_list(cls, expectations):
        new_block = ExpectationSuiteBulletListContentBlockRenderer.render(
            expectations,
            include_column_name=False,
        )

        return [], new_block

    @classmethod
    def render(cls, expectations={}):
        column = cls._get_column_name(expectations)

        content_blocks = []
        remaining_expectations, header_block = cls._render_header(expectations)
        content_blocks.append(header_block)
        # remaining_expectations, content_blocks = cls._render_column_type(
        # remaining_expectations, content_blocks)
        remaining_expectations, bullet_block = cls._render_bullet_list(remaining_expectations)
        content_blocks.append(bullet_block)

        # NOTE : Some render* functions return None so we filter them out
        populated_content_blocks = list(filter(None, content_blocks))
        return RenderedSectionContent(**{
            "section_name": column,
            "content_blocks": populated_content_blocks
        })


class MultiBatchMetricsColumnSectionRenderer(ColumnSectionRenderer):
    metric_render_mapper = {
        "column_min": ["basic_histogram", "line_chart"],
        "column_max": ["basic_histogram", "line_chart"],
        "column_mean": ["basic_histogram", "line_chart"],
        "column_stdev": ["basic_histogram", "line_chart"],
        "column_median": ["basic_histogram", "line_chart"],
        "column_proportion_of_unique_values": ["basic_histogram", "line_chart"],
        "column_unique_count": ["basic_histogram", "line_chart"],
        "row_count": ["basic_histogram", "line_chart"],
        "column_quantiles": ["quantile_line_chart", "quantile_histogram"],
        "distinct_set_members": ["distinct_set_member_matrix"]
    }
    
    @classmethod
    def render(cls, multi_batch_metrics_dicts, column_name=None):
        section_name = column_name if column_name else "Non-Column Metrics"
        content_blocks = [cls._render_header(section_name)]
        
        for metric_dict in multi_batch_metrics_dicts:
            content_blocks += cls._render_metric_blocks(metric_dict)
        
        return RenderedSectionContent(**{
            "section_name": section_name,
            "content_blocks": content_blocks
        })
    
    @classmethod
    def _render_metric_blocks(cls, metric_dict):
        metric_name = metric_dict.get("metric_name")
        expectation_type = metric_dict["expectation_type"]
        
        if not metric_name or not cls.metric_render_mapper.get(metric_name):
            return []
        
        metric_block_names = cls.metric_render_mapper.get(metric_name)
        
        metric_blocks = [cls._render_metric_heading(metric_name, expectation_type)]
        
        for block_name in metric_block_names:
            block_renderer = getattr(cls, "_render_{block_name}".format(block_name=block_name), None)
            if block_renderer:
                metric_block = getattr(cls, "_render_{block_name}".format(block_name=block_name))(metric_dict)
                if type(metric_block) is list:
                    metric_blocks += metric_block
                else:
                    metric_blocks.append(metric_block)
        
        return metric_blocks
    
    @classmethod
    def _render_metric_heading(cls, metric_name, expectation_type):
        return RenderedComponentContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "tag": "div",
                "template": "$metric_name",
                "params": {
                  "metric_name": metric_name
                },
                "tooltip": {
                    "content": expectation_type
                },
                "styling": {
                    "params": {
                        "metric_name": {
                            "tag": "h5",
                        }
                    },
                    "classes": [
                        "col-12",
                        "pt-2",
                        "pb-1",
                        "pl-4",
                        "border-top",
                        "border-bottom",
                        "border-dark",
                    ]
                }
            },
            "styling": {
                "classes": [
                    "col-12",
                    "mt-3",
                    "mb-2",
                ]
            }
        })
    
    @classmethod
    def _render_distinct_set_member_matrix(cls, metric_dict):
        batch_indices = [batch_fingerprint.split("__")[0] for batch_fingerprint in metric_dict["batch_fingerprints"]]
        metric_values = metric_dict["batch_metric_values"]
        value_sets_union = list(set(reduce(lambda x1, x2: x1 + x2, metric_values)))
        
        df_indices = []
        value_set_member_list = []
        has_value_set_member_list = []

        for list_idx, batch_index in enumerate(batch_indices):
            for value_set_member in value_sets_union:
                df_indices.append(batch_index)
                value_set_member_list.append(value_set_member)
                has_value_set_member_list.append(
                    True) if value_set_member in metric_values[list_idx] else has_value_set_member_list.append(False)

        value_sets_dict = {
            'batch_index': df_indices,
            'value_set_member': value_set_member_list,
            'has_value_set_member': has_value_set_member_list
        }

        value_sets_df = pd.DataFrame(value_sets_dict)

        distinct_set_member_matrix = alt.Chart(value_sets_df).mark_rect(stroke='black').encode(
            x='batch_index:O',
            y='value_set_member:N',
            color='has_value_set_member:N'
        )
        distinct_set_member_matrix_json = distinct_set_member_matrix.to_json()
        
        return RenderedComponentContent(**{
            "content_block_type": "graph",
            "graph": distinct_set_member_matrix_json,
            "styling": {
                "classes": ["col-12"],
                "styles": {
                    "margin-top": "20px"
                }
            }
        })
    
    @classmethod
    def _render_basic_histogram(cls, metric_dict):
        batch_indices = [batch_fingerprint.split("__")[0] for batch_fingerprint in metric_dict["batch_fingerprints"]]
        metric_values = metric_dict["batch_metric_values"]
        
        histogram_df = pd.DataFrame({
            "batch_index": batch_indices,
            "metric_value": metric_values
        })
        
        histogram = alt.Chart(histogram_df).mark_bar().encode(
            alt.X('metric_value', bin=True),
            y='count()'
        ).properties(
            width=300,
            height=300,
            autosize="fit",
            title="Metric Histogram"
        )
        
        histogram_json = histogram.to_json()
        
        return RenderedComponentContent(**{
            "content_block_type": "graph",
            "graph": histogram_json,
            "styling": {
                "classes": ["col-4"],
                "styles": {
                    "margin-top": "20px"
                }
            }
        })
    
    @classmethod
    def _render_line_chart(cls, metric_dict):
        batch_indices = [batch_fingerprint.split("__")[0] for batch_fingerprint in metric_dict["batch_fingerprints"]]
        metric_values = metric_dict["batch_metric_values"]
    
        line_chart_df = pd.DataFrame({
            "batch_index": batch_indices,
            "metric_value": metric_values
        })
    
        line_chart = alt.Chart(line_chart_df).mark_line(point=True).encode(
            alt.X('batch_index:O'),
            y='metric_value'
        ).properties(
            width=650,
            height=300,
            autosize="fit",
            title="Metric Values Across Batches"
        )
    
        line_chart_json = line_chart.to_json()
    
        return RenderedComponentContent(**{
            "content_block_type": "graph",
            "graph": line_chart_json,
            "styling": {
                "classes": ["col-8"],
                "styles": {
                    "margin-top": "20px"
                }
            }
        })
    
    @classmethod
    def _render_quantile_histogram(cls, metric_dict):
        batch_indices = [batch_fingerprint.split("__")[0] for batch_fingerprint in metric_dict["batch_fingerprints"]]
        metric_values = metric_dict["batch_metric_values"]
        quantiles = metric_values[0]["quantiles"]
        quantile_values = [metric_value["values"] for metric_value in metric_values]
        
        quantiles_hist_dict = {
            # "batch_indices": batch_indices
        }

        quantile_strings = {
            .25: "Q1",
            .75: "Q3",
            .50: "Median"
        }
        
        for list_idx, batch_idx in enumerate(batch_indices):
            for quantile_idx, quantile in enumerate(quantiles):
                quantile_key = quantile_strings.get(
                    quantile,
                    str(int(quantile * 100)) + "%"
                )
                if not quantiles_hist_dict.get(quantile_key):
                    quantiles_hist_dict[quantile_key] = []
                quantiles_hist_dict[quantile_key].append(quantile_values[list_idx][quantile_idx])

        quantiles_hist_df = pd.DataFrame(quantiles_hist_dict)

        quantiles_histograms = []

        for quantile in quantiles_hist_df.columns:
            quantiles_histograms.append(
                alt.Chart(quantiles_hist_df).mark_bar().encode(
                    alt.X(quantile, bin=True),
                    y='count()'
                ).properties(
                    width=300,
                    height=300,
                    autosize="fit",
                    title="{quantile} Quantile Histogram".format(quantile=quantile)
                )
            )
            
        quantiles_histograms_json = [histogram.to_json() for histogram in  quantiles_histograms]
        
        return [
            RenderedComponentContent(**{
                "content_block_type": "graph",
                "graph": histogram_json,
                "styling": {
                    "classes": ["col-4"],
                    "styles": {
                        "margin-top": "20px"
                    }
                }
            }) for histogram_json in quantiles_histograms_json
        ]
    
    @classmethod
    def _render_quantile_line_chart(cls, metric_dict):
        batch_indices = [batch_fingerprint.split("__")[0] for batch_fingerprint in metric_dict["batch_fingerprints"]]
        metric_values = metric_dict["batch_metric_values"]
        quantiles = metric_values[0]["quantiles"]
        quantile_values = [metric_value["values"] for metric_value in metric_values]
        
        quantile_line_chart_dict = {
            "batch_index": [],
            "quantile": [],
            "value": []
        }

        quantile_strings = {
            .25: "Q1",
            .75: "Q3",
            .50: "Median"
        }
        
        for list_idx, batch_idx in enumerate(batch_indices):
            for quantile_idx, quantile in enumerate(quantiles):
                quantile_key = quantile_strings.get(
                    quantile,
                    str(int(quantile * 100)) + "%"
                )
                quantile_line_chart_dict["quantile"].append(quantile_key)
                quantile_line_chart_dict["batch_index"].append(batch_idx)
                quantile_line_chart_dict["value"].append(quantile_values[list_idx][quantile_idx])
                
        quantile_line_chart_df = pd.DataFrame(quantile_line_chart_dict)

        nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['batch_index'], empty='none')

        line = alt.Chart(quantile_line_chart_df).mark_line().encode(
            x='batch_index:O',
            y='value:Q',
            color='quantile:N'
        )

        selectors = alt.Chart(quantile_line_chart_df).mark_point().encode(
            x='batch_index:O',
            opacity=alt.value(0)
        ).add_selection(nearest)

        points = line.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        text = line.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'value:Q', alt.value(' '))
        )

        rules = alt.Chart(quantile_line_chart_df).mark_rule(color='gray').encode(
            x='batch_index:O'
        ).transform_filter(nearest)

        quantile_line_chart = alt.layer(
            line, selectors, points, rules, text
        ).properties(
            width=950,
            height=400,
            title="Quantile Values Across Batches"
        )
        quantile_line_chart_json = quantile_line_chart.to_json()
        
        return RenderedComponentContent(**{
            "content_block_type": "graph",
            "graph": quantile_line_chart_json,
            "styling": {
                "classes": ["col-12"],
                "styles": {
                    "margin-top": "20px",
                    "margin-bottom": "20px"
                }
            }
        })
    
    @classmethod
    def _render_header(cls, header):
        return RenderedComponentContent(**{
            "content_block_type": "header",
            "header": {
                "template": convert_to_string_and_escape(header),
                "tooltip": {
                    "content": "expect_column_to_exist",
                    "placement": "top"
                }
            },
            "styling": {
                "classes": ["col-12"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })
    