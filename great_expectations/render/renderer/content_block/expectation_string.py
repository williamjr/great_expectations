# -*- coding: utf-8 -*-
import copy
import json

from great_expectations.render.renderer.content_block.content_block import ContentBlockRenderer
from great_expectations.render.util import ordinal, num_to_str

import pandas as pd
import altair as alt


def substitute_none_for_missing(kwargs, kwarg_list):
    """Utility function to plug Nones in when optional parameters are not specified in expectation kwargs.

    Example:
        Input:
            kwargs={"a":1, "b":2},
            kwarg_list=["c", "d"]

        Output: {"a":1, "b":2, "c": None, "d": None}

    This is helpful for standardizing the input objects for rendering functions.
    The alternative is lots of awkward `if "some_param" not in kwargs or kwargs["some_param"] == None:` clauses in renderers.
    """

    new_kwargs = copy.deepcopy(kwargs)
    for kwarg in kwarg_list:
        if kwarg not in new_kwargs:
            new_kwargs[kwarg] = None
    return new_kwargs


class ExpectationStringRenderer(ContentBlockRenderer):

    @classmethod
    def _missing_content_block_fn(cls, expectation, styling=None, include_column_name=True):
        return [{
            "content_block_type": "string_template",
            "styling": {
              "parent": {
                  "classes": ["alert", "alert-warning"]
              }
            },
            "string_template": {
                "template": "$expectation_type(**$kwargs)",
                "params": {
                    "expectation_type": expectation["expectation_type"],
                    "kwargs": expectation["kwargs"]
                },
                "styling": {
                    "params": {
                        "expectation_type": {
                            "classes": ["badge", "badge-warning"],
                        }
                    }
                },
            }
        }]
    
    @classmethod
    def expect_column_to_exist(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "column_index"],
        )
        
        if params["column_index"] is None:
            if include_column_name:
                template_str = "$column is a required field."
            else:
                template_str = "is a required field."
        else:
            params["column_indexth"] = ordinal(params["column_index"])
            if include_column_name:
                template_str = "$column must be the $column_indexth field"
            else:
                template_str = "must be the $column_indexth field"
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_unique_value_count_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value", "mostly"],
        )
        
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "may have any number of unique values."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if params["min_value"] is None:
                    template_str = "must have fewer than $max_value unique values, at least $mostly_pct % of the time."
                elif params["max_value"] is None:
                    template_str = "must have more than $min_value unique values, at least $mostly_pct % of the time."
                else:
                    template_str = "must have between $min_value and $max_value unique values, at least $mostly_pct % of the time."
            else:
                if params["min_value"] is None:
                    template_str = "must have fewer than $max_value unique values."
                elif params["max_value"] is None:
                    template_str = "must have more than $min_value unique values."
                else:
                    template_str = "must have between $min_value and $max_value unique values."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    # NOTE: This method is a pretty good example of good usage of `params`.
    @classmethod
    def expect_column_values_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value", "mostly"]
        )
        
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "may have any numerical value."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must be between $min_value and $max_value, at least $mostly_pct % of the time."
                
                elif params["min_value"] is None:
                    template_str = "values must be less than $max_value, at least $mostly_pct % of the time."
                
                elif params["max_value"] is None:
                    template_str = "values must be less than $max_value, at least $mostly_pct % of the time."
            else:
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must always be between $min_value and $max_value."
                
                elif params["min_value"] is None:
                    template_str = "values must always be less than $max_value."
                
                elif params["max_value"] is None:
                    template_str = "values must always be more than $min_value."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_pair_values_A_to_be_greater_than_B(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column_A", "column_B", "parse_strings_as_datetimes",
             "ignore_row_if", "mostly", "or_equal"]
        )
        
        if (params["column_A"] is None) or (params["column_B"] is None):
            template_str = "$column has a bogus `expect_column_pair_values_A_to_be_greater_than_B` expectation."
        
        if params["mostly"] is None:
            if params["or_equal"] in [None, False]:
                template_str = "Values in $column_A must always be greater than those in $column_B."
            else:
                template_str = "Values in $column_A must always be greater than or equal to those in $column_B."
        else:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            if params["or_equal"] in [None, False]:
                template_str = "Values in $column_A must be greater than those in $column_B, at least $mostly_pct % of the time."
            else:
                template_str = "Values in $column_A must be greater than or equal to those in $column_B, at least $mostly_pct % of the time."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_pair_values_to_be_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column_A", "column_B",
             "ignore_row_if", "mostly", ]
        )
        
        # NOTE: This renderer doesn't do anything with "ignore_row_if"
        
        if (params["column_A"] is None) or (params["column_B"] is None):
            template_str = " unrecognized kwargs for expect_column_pair_values_to_be_equal: missing column."
        
        if params["mostly"] is None:
            template_str = "Values in $column_A and $column_B must always be equal."
        else:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str = "Values in $column_A and $column_B must be equal, at least $mostly_pct % of the time."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_table_columns_to_match_ordered_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column_list"]
        )
        
        if params["column_list"] is None:
            template_str = "This table should have a list of columns in a specific order, but that order is not specified."
        
        else:
            template_str = "This table should have these columns in this order: "
            for idx in range(len(params["column_list"]) - 1):
                template_str += "$column_list_" + str(idx) + ", "
                params["column_list_" + str(idx)] = params["column_list"][idx]
            
            last_idx = len(params["column_list"]) - 1
            template_str += "$column_list_" + str(last_idx)
            params["column_list_" + str(last_idx)] = params["column_list"][last_idx]
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_multicolumn_values_to_be_unique(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column_list", "ignore_row_if"]
        )
        
        template_str = "Values must always be unique across columns: "
        for idx in range(len(params["column_list"]) - 1):
            template_str += "$column_list_" + str(idx) + ", "
            params["column_list_" + str(idx)] = params["column_list"][idx]
        
        last_idx = len(params["column_list"]) - 1
        template_str += "$column_list_" + str(last_idx)
        params["column_list_" + str(last_idx)] = params["column_list"][last_idx]
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]

    @classmethod
    def expect_table_column_count_to_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["value"]
        )
        template_str = "Must have exactly $value columns."
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]

    @classmethod
    def expect_table_column_count_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["min_value", "max_value"]
        )
        if params["min_value"] is None and params["max_value"] is None:
            template_str = "May have any number of columns."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "Must have between $min_value and $max_value columns."
            elif params["min_value"] is None:
                template_str = "Must have less than than $max_value columns."
            elif params["max_value"] is None:
                template_str = "Must have more than $min_value columns."
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_table_row_count_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["min_value", "max_value"]
        )
        
        if params["min_value"] is None and params["max_value"] is None:
            template_str = "May have any number of rows."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "Must have between $min_value and $max_value rows."
            elif params["min_value"] is None:
                template_str = "Must have less than than $max_value rows."
            elif params["max_value"] is None:
                template_str = "Must have more than $min_value rows."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_table_row_count_to_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["value"]
        )
        template_str = "Must have exactly $value rows."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_distinct_values_to_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "value_set"],
        )
        
        if params["value_set"] is None or len(params["value_set"]) == 0:
            
            if include_column_name:
                template_str = "$column distinct values must belong to this set: [ ]"
            else:
                template_str = "distinct values must belong to a set, but that set is not specified."
        
        else:
            
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )
            
            if include_column_name:
                template_str = "$column distinct values must belong to this set: " + values_string + "."
            else:
                template_str = "distinct values must belong to this set: " + values_string + "."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_not_be_null(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "mostly"],
        )

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            if include_column_name:
                template_str = "$column values must not be null, at least $mostly_pct % of the time."
            else:
                template_str = "values must not be null, at least $mostly_pct % of the time."
        else:
            if include_column_name:
                template_str = "$column values must never be null."
            else:
                template_str = "values must never be null."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_be_null(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "mostly"]
        )
        
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str = "values must be null, at least $mostly_pct % of the time."
        else:
            template_str = "values must be null."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_be_of_type(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "type_", "mostly"]
        )
        
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str = "values must be of type $type_, at least $mostly_pct % of the time."
        else:
            template_str = "values must be of type $type_."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_be_in_type_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "type_list", "mostly"],
        )

        if params["type_list"] is not None:
            for i, v in enumerate(params["type_list"]):
                params["v__"+str(i)] = v
            values_string = " ".join(
                ["$v__"+str(i) for i, v in enumerate(params["type_list"])]
            )

            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if include_column_name:
                    template_str = "$column value types must belong to this set: " + values_string + ", at least $mostly_pct % of the time."
                else:
                    template_str = "value types must belong to this set: " + values_string + ", at least $mostly_pct % of the time."
            else:
                if include_column_name:
                    template_str = "$column value types must belong to this set: "+values_string+"."
                else:
                    template_str = "value types must belong to this set: "+values_string+"."
        else:
            if include_column_name:
                template_str = "$column value types may be any value, but observed value will be reported"
            else:
                template_str = "value types may be any value, but observed value will be reported"

        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]

    @classmethod
    def expect_column_values_to_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "value_set", "mostly", "parse_strings_as_datetimes"]
        )
        
        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v
            
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )
            
        template_str = "values must belong to this set: " + values_string
        
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_not_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "value_set", "mostly", "parse_strings_as_datetimes"]
        )
        
        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v
            
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )
            
        template_str = "values must not belong to this set: " + values_string
    
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        if include_column_name:
            template_str = "$column"
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_proportion_of_unique_values_to_be_between(cls, expectation, styling=None,
                                                                include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value"],
        )
        
        if params["min_value"] is None and params["max_value"] is None:
            template_str = "may have any fraction of unique values."
        else:
            if params["min_value"] is None:
                template_str = "fraction of unique values must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "fraction of unique values must be at least $min_value."
            else:
                template_str = "fraction of unique values must be between $min_value and $max_value."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    # TODO: test parse_strings_as_datetimes
    @classmethod
    def expect_column_values_to_be_increasing(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "strictly", "mostly", "parse_strings_as_datetimes"]
        )
        
        if params.get("strictly"):
            template_str = "values must be strictly greater than previous values"
        else:
            template_str = "values must be greater than or equal to previous values"
            
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    # TODO: test parse_strings_as_datetimes
    @classmethod
    def expect_column_values_to_be_decreasing(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "strictly", "mostly", "parse_strings_as_datetimes"]
        )
        
        if params.get("strictly"):
            template_str = "values must be strictly less than previous values"
        else:
            template_str = "values must be less than or equal to previous values"
    
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_value_lengths_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value", "mostly"],
        )
        
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "values may have any length."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must be between $min_value and $max_value characters long, at least $mostly_pct % of the time."
                
                elif params["min_value"] is None:
                    template_str = "values must be less than $max_value characters long, at least $mostly_pct % of the time."
                
                elif params["max_value"] is None:
                    template_str = "values must be more than $min_value characters long, at least $mostly_pct % of the time."
            else:
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must always be between $min_value and $max_value characters long."
                
                elif params["min_value"] is None:
                    template_str = "values must always be less than $max_value characters long."
                
                elif params["max_value"] is None:
                    template_str = "values must always be more than $min_value characters long."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_value_lengths_to_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "value", "mostly"]
        )
        
        if params.get("value") is None:
            template_str = "values may have any length."
        else:
            template_str = "values must be $value characters long"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str += ", at least $mostly_pct % of the time."
            else:
                template_str += "."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_match_regex(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "regex", "mostly"]
        )
        
        if not params.get("regex"):
            template_str = "values must match a regular expression but none was specified."
        else:
            template_str = "values must match this regular expression: $regex"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str += ", at least $mostly_pct % of the time."
            else:
                template_str += "."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_not_match_regex(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "regex", "mostly"],
        )
        
        if not params.get("regex"):
            template_str = "values must not match a regular expression but none was specified."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if include_column_name:
                    template_str = "$column values must not match this regular expression: $regex, at least $mostly_pct % of the time."
                else:
                    template_str = "values must not match this regular expression: $regex, at least $mostly_pct % of the time."
            else:
                if include_column_name:
                    template_str = "$column values must not match this regular expression: $regex."
                else:
                    template_str = "values must not match this regular expression: $regex."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_match_regex_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "regex_list", "mostly", "match_on"],
        )
        
        if not params.get("regex_list") or len(params.get("regex_list")) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["regex_list"]):
                params["v__" + str(i)] = v
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["regex_list"])]
            )
            
        if params.get("match_on") == "all":
            template_str = "values must match all of the following regular expressions: " + values_string
        else:
            template_str = "values must match any of the following regular expressions: " + values_string
            
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_not_match_regex_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "regex_list", "mostly"],
        )
        
        if not params.get("regex_list") or len(params.get("regex_list")) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["regex_list"]):
                params["v__" + str(i)] = v
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["regex_list"])]
            )
            
        template_str = "values must not match any of the following regular expressions: " + values_string
        
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_match_strftime_format(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "strftime_format", "mostly"],
        )
        
        if not params.get("strftime_format"):
            template_str = "values must match a strftime format but none was specified."
        else:
            template_str = "values must match the following strftime format: $strftime_format"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str += ", at least $mostly_pct % of the time."
            else:
                template_str += "."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_be_dateutil_parseable(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "mostly"],
        )
        
        template_str = "values must be parseable by dateutil"
        
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_be_json_parseable(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "mostly"],
        )
        
        template_str = "values must be parseable as JSON"
    
        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_values_to_match_json_schema(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "mostly", "json_schema"],
        )
        
        if not params.get("json_schema"):
            template_str = "values must match a JSON Schema but none was specified."
        else:
            params["formatted_json"] = "<pre>" + json.dumps(params.get("json_schema"), indent=4) + "</pre>"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str = "values must match the following JSON Schema, at least $mostly_pct % of the time: $formatted_json"
            else:
                template_str = "values must match the following JSON Schema: $formatted_json"
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": {
                    "params":
                        {
                            "formatted_json": {
                                "classes": []
                            }
                        }
                },
            }
        }]
    
    @classmethod
    def expect_column_distinct_values_to_contain_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "value_set", "parse_strings_as_datetimes"]
        )
        
        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v
            
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )
            
        template_str = "distinct values must contain this set: " + values_string + "."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_distinct_values_to_equal_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "value_set", "parse_strings_as_datetimes"]
        )
        
        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v
            
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )
            
        template_str = "distinct values must match this set: " + values_string + "."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_mean_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value"]
        )
        
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "mean may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "mean must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "mean must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "mean must be more than $min_value."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]

    @classmethod
    def expect_column_median_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value"]
        )
    
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "median may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "median must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "median must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "median must be more than $min_value."
    
        if include_column_name:
            template_str = "$column " + template_str
    
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
        
    @classmethod
    def expect_column_stdev_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value"]
        )
    
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "standard deviation may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "standard deviation must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "standard deviation must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "standard deviation must be more than $min_value."
    
        if include_column_name:
            template_str = "$column " + template_str
    
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
        
    @classmethod
    def expect_column_max_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value", "parse_strings_as_datetimes"]
        )
    
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "maximum value may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "maximum value must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "maximum value must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "maximum value must be more than $min_value."
    
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
    
        if include_column_name:
            template_str = "$column " + template_str
    
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_min_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value", "parse_strings_as_datetimes"]
        )
        
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "minimum value may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "minimum value must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "minimum value must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "minimum value must be more than $min_value."
        
        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_sum_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "min_value", "max_value"]
        )
        
        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "sum may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "sum must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "sum must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "sum must be more than $min_value."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_most_common_value_to_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "value_set", "ties_okay"]
        )
        
        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v
            
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )
            
        template_str = "most common value must belong to this set: " + values_string + "."
        
        if params.get("ties_okay"):
            template_str += " Values outside this set that are as common (but not more common) are allowed."
        
        if include_column_name:
            template_str = "$column " + template_str
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
    
    @classmethod
    def expect_column_quantile_values_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "quantile_ranges"]
        )
        template_str = "Column quantiles must be within the following value ranges:\n\n"

        if include_column_name:
            template_str = "$column " + template_str

        expectation_string_obj = {
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params
            }
        }

        quantiles = params["quantile_ranges"]["quantiles"]
        value_ranges = params["quantile_ranges"]["value_ranges"]

        table_header_row = ["Quantile", "Min Value", "Max Value"]
        table_rows = []

        quantile_strings = {
            .25: "Q1",
            .75: "Q3",
            .50: "Median"
        }

        for idx, quantile in enumerate(quantiles):
            quantile_string = quantile_strings.get(quantile)
            table_rows.append([
                quantile_string if quantile_string else "{:3.2f}".format(quantile),
                str(value_ranges[idx][0]) if value_ranges[idx][0] else "Any",
                str(value_ranges[idx][1]) if value_ranges[idx][1] else "Any",
            ])

        quantile_range_table = {
            "content_block_type": "table",
            "header_row": table_header_row,
            "table": table_rows,
            "styling": {
                "body": {
                    "classes": ["table", "table-sm", "table-unbordered", "col-4"],
                },
                "parent": {
                    "styles": {
                        "list-style-type": "none"
                    }
                }
            }
        }

        return [
            expectation_string_obj,
            quantile_range_table
        ]

    @classmethod
    def expect_column_kl_divergence_to_be_less_than(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "partition_object", "threshold"]
        )
        
        expected_distribution = None
        if not params.get("partition_object"):
            template_str = "Column can match any distribution."
        else:
            template_str = "Kullback-Leibler (KL) divergence with respect to the following distribution must be " \
                           "lower than $threshold:\n\n"

            weights = params["partition_object"]["weights"]
            if len(weights) <= 10:
                height = 200
                width = 200
                col_width = 4
            else:
                height = 300
                width = 300
                col_width = 6
                
            if params["partition_object"].get("bins"):
                bins = params["partition_object"]["bins"]
                bins_x1 = [round(value, 1) for value in bins[:-1]]
                bins_x2 = [round(value, 1) for value in bins[1:]]
    
                df = pd.DataFrame({
                    "bin_min": bins_x1,
                    "bin_max": bins_x2,
                    "fraction": weights,
                })

                bars = alt.Chart(df).mark_bar().encode(
                    x='bin_min:O',
                    x2='bin_max:O',
                    y="fraction:Q"
                ).properties(width=width, height=height, autosize="fit")
    
                chart = bars.to_json()
            elif params["partition_object"].get("values"):
                values = params["partition_object"]["values"]
                
                df = pd.DataFrame({
                    "values": values,
                    "fraction": weights
                })

                bars = alt.Chart(df).mark_bar().encode(
                    x='values:N',
                    y="fraction:Q"
                ).properties(width=width, height=height, autosize="fit")
                chart = bars.to_json()

            expected_distribution = {
                "content_block_type": "graph",
                "graph": chart,
                "styling": {
                    "classes": ["col-" + str(col_width)],
                    "styles": {
                        "margin-top": "20px",
                    },
                    "parent": {
                        "styles": {
                            "list-style-type": "none"
                        }
                    }
                }
            }

        if include_column_name:
            template_str = "$column " + template_str
        
        expectation_string_obj = {
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params
            }
        }
        
        if expected_distribution:
            return [
                expectation_string_obj,
                expected_distribution
            ]
        else:
            return [expectation_string_obj]
    
    @classmethod
    def expect_column_values_to_be_unique(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "mostly"],
        )

        if include_column_name:
            template_str = "$column values must be unique"
        else:
            template_str = "values must be unique"

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."
        
        return [{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        }]
