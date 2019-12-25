import datetime
import logging
logger = logging.getLogger(__name__)

from six import string_types

from great_expectations.data_context.util import (
    instantiate_class_from_config,
)
from ..data_context.types import (
    DataAssetIdentifier,
    ValidationResultIdentifier,
    ExpectationSuiteIdentifier,
)
from great_expectations.data_asset import (
    DataAsset,
)
from .util import send_slack_notification

# NOTE: Abe 2019/08/24 : This is first implementation of all these classes. Consider them UNSTABLE for now. 

class ValidationOperator(object):
    """
    The base class of all validation operators.

    It defines the signature of the public run method - this is the only
    contract re operators' API. Everything else is up to the implementors
    of validation operator classes that will be the descendants of this base class.
    """

    def run(self, assets_to_validate, run_id):
        raise NotImplementedError


class ActionListValidationOperator(ValidationOperator):
    """
    ActionListValidationOperator is a validation operator
    that validates each batch in the list that is passed to its run
    method and then invokes a list of configured actions on every
    validation result.

    A user can configure the list of actions to invoke.

    Each action in the list must be an instance of NamespacedValidationAction
    class (or its descendants).

    Below is an example of this operator's configuration::

        action_list_operator:
            class_name: ActionListValidationOperator
            action_list:
              - name: store_validation_result
                action:
                  class_name: StoreAction
                  target_store_name: validations_store
              - name: store_evaluation_params
                action:
                  class_name: ExtractAndStoreEvaluationParamsAction
                  target_store_name: evaluation_parameter_store
              - name: send_slack_notification_on_validation_result
                action:
                  class_name: SlackNotificationAction
                  # put the actual webhook URL in the uncommitted/config_variables.yml file
                  slack_webhook: ${validation_notification_slack_webhook}
                 notify_on: all # possible values: "all", "failure", "success"
                  renderer:
                    module_name: great_expectations.render.renderer.slack_renderer
                    class_name: SlackRenderer
    """

    def __init__(self, data_context, action_list):
        self.data_context = data_context

        self.action_list = action_list
        self.actions = {}
        for action_config in action_list:
            assert isinstance(action_config, dict)
            #NOTE: Eugene: 2019-09-23: need a better way to validate an action config:
            if not set(action_config.keys()) == {"name", "action"}:
                raise KeyError('Action config keys must be ("name", "action"). Instead got {}'.format(action_config.keys()))

            new_action = instantiate_class_from_config(
                config=action_config["action"],
                runtime_config={
                    "data_context": self.data_context,
                },
                config_defaults={
                    "module_name": "great_expectations.validation_operators"
                }
            )
            self.actions[action_config["name"]] = new_action

    def _build_batch_from_item(self, item):
        """Internal helper method to take an asset to validate, which can be either:
          (1) a DataAsset; or
          (2) a tuple of data_asset_name, expectation_suite_name, and batch_kwargs (suitable for passing to get_batch)

        Args:
            item: The item to convert to a batch (see above)

        Returns:
            A batch of data

        """
        if not isinstance(item, DataAsset):
            batch = self.data_context.get_batch(
                data_asset_name=item[0],
                expectation_suite_name=item[1],
                batch_kwargs=item[2]
            )
        else:
            batch = item

        return batch

    def run(self, assets_to_validate, run_id):
        result_object = {
            "success": None,
            "details": {}
        }

        for item in assets_to_validate:
            batch = self._build_batch_from_item(item)
            expectation_suite_identifier = ExpectationSuiteIdentifier(
                data_asset_name=DataAssetIdentifier(
                    *self.data_context.normalize_data_asset_name(batch._expectation_suite["data_asset_name"])
                ),
                expectation_suite_name=batch._expectation_suite.expectation_suite_name
            )
            validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=expectation_suite_identifier,
                run_id=run_id,
            )
            result_object["details"][expectation_suite_identifier] = {}
            batch_validation_result = batch.validate(run_id=run_id, result_format="SUMMARY")
            result_object["details"][expectation_suite_identifier]["validation_result"] = batch_validation_result
            batch_actions_results = self._run_actions(batch, expectation_suite_identifier, batch._expectation_suite, batch_validation_result, run_id)
            result_object["details"][expectation_suite_identifier]["actions_results"] = batch_actions_results

        result_object["success"] = all([val["validation_result"]["success"] for val in result_object["details"].values()])

        return result_object

    def _run_actions(self, batch, expectation_suite_identifier, expectation_suite, batch_validation_result, run_id):
        """
        Runs all actions configured for this operator on the result of validating one
        batch against one expectation suite.

        If an action fails with an exception, the method does not continue.

        :param batch:
        :param expectation_suite:
        :param batch_validation_result:
        :param run_id:
        :return: a dictionary: {action name -> result returned by the action}
        """
        batch_actions_results = {}
        for action in self.action_list:
            # NOTE: Eugene: 2019-09-23: log the info about the batch and the expectation suite
            logger.debug("Processing validation action with name {}".format(action["name"]))

            validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=expectation_suite_identifier,
                run_id=run_id,
            )
            try:
                action_result = self.actions[action["name"]].run(
                                                validation_result_suite_identifier=validation_result_id,
                                                validation_result_suite=batch_validation_result,
                                                data_asset=batch
                )

                batch_actions_results[action["name"]] = {} if action_result is None else action_result
            except Exception as e:
                logger.exception("Error running action with name {}".format(action["name"]))
                raise e

        return batch_actions_results



        result_object = {}

        for item in assets_to_validate:
            batch = self._build_batch_from_item(item)
            expectation_suite_identifier = ExpectationSuiteIdentifier(
                data_asset_name=batch.data_asset_identifier,
                expectation_suite_name=batch._expectation_suite.expectation_suite_name
            )
            validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=expectation_suite_identifier,
                run_id=run_id,
            )
            result_object[validation_result_id] = {}
            batch_validation_result = batch.validate(result_format="SUMMARY")
            result_object[validation_result_id]["validation_result"] = batch_validation_result
            batch_actions_results = self._run_actions(batch, batch._expectation_suite, batch_validation_result, run_id)
            result_object[validation_result_id]["actions_results"] = batch_actions_results

        # NOTE: Eugene: 2019-09-24: Need to define this result object. Discussion required!
        return result_object


class WarningAndFailureExpectationSuitesValidationOperator(ActionListValidationOperator):
    """WarningAndFailureExpectationSuitesValidationOperator is a validation operator
    that accepts a list batches of data assets (or the information necessary to fetch these batches).
    The operator retrieves 2 expectation suites for each data asset/batch - one containing
    the critical expectations ("failure") and the other containing non-critical expectations
    ("warning"). By default, the operator assumes that the first is called "failure" and the
    second is called "warning", but "expectation_suite_name_prefix" attribute can be specified
    in the operator's configuration to make sure it searched for "{expectation_suite_name_prefix}failure"
    and {expectation_suite_name_prefix}warning" expectation suites for each data asset.

    The operator validates each batch against its "failure" and "warning" expectation suites and
    invokes a list of actions on every validation result.

    The list of these actions is specified in the operator's configuration

    Each action in the list must be an instance of NamespacedValidationAction
    class (or its descendants).

    The operator sends a Slack notification (if "slack_webhook" is present in its
    config). The "notify_on" config property controls whether the notification
    should be sent only in the case of failure ("failure"), only in the case
    of success ("success"), or always ("all").

    Below is an example of this operator's configuration::


        run_warning_and_failure_expectation_suites:
            class_name: WarningAndFailureExpectationSuitesValidationOperator
            # put the actual webhook URL in the uncommitted/config_variables.yml file
            slack_webhook: ${validation_notification_slack_webhook}
            action_list:
              - name: store_validation_result
                action:
                  class_name: StoreAction
                  target_store_name: validations_store
              - name: store_evaluation_params
                action:
                  class_name: ExtractAndStoreEvaluationParamsAction
                  target_store_name: evaluation_parameter_store


    The operator returns an object that looks like the example below.

    The value of "success" is True if no critical expectation suites ("failure")
    failed to validate (non-critial ("warning") expectation suites
    are allowed to fail without affecting the success status of the run::


        {
            "data_asset_identifiers": [list, of, data, asset, identifiers],
            "success": True/False,
            "failure": {
                "expectation_suite_identifier": {
                    "validation_result": validation_result,
                    "action_results": {
                        "action name": "action result object"
                    }
                }
            },
            "warning": {
                "expectation_suite_identifier": {
                    "validation_result": validation_result,
                    "action_results": {
                        "action name": "action result object"
                    }
                }
            }
        }

    """


    def __init__(self,
        data_context,
        action_list,
        expectation_suite_name_prefix="",
        expectation_suite_name_suffixes=["failure", "warning"],
        stop_on_first_error=False,
        slack_webhook=None,
        notify_on="all"
    ):
        super(WarningAndFailureExpectationSuitesValidationOperator, self).__init__(
            data_context,
            action_list,
        )

        self.stop_on_first_error = stop_on_first_error
        self.expectation_suite_name_prefix = expectation_suite_name_prefix

        assert len(expectation_suite_name_suffixes) == 2
        for suffix in expectation_suite_name_suffixes:
            assert isinstance(suffix, string_types)
        self.expectation_suite_name_suffixes = expectation_suite_name_suffixes
        
        self.slack_webhook = slack_webhook
        self.notify_on = notify_on

    def _build_slack_query(self, run_return_obj):
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%x %X")
        success = run_return_obj.get("success")
        status_text = "Success :tada:" if success else "Failed :x:"
        run_id = run_return_obj.get("run_id")
        data_asset_identifiers = run_return_obj.get("data_asset_identifiers")
        failed_data_assets = []
        
        if run_return_obj.get("failure"):
            failed_data_assets = [
                validation_result_identifier["expectation_suite_identifier"]["data_asset_name"] for validation_result_identifier, value in run_return_obj.get("failure").items() \
                if not value["validation_result"]["success"]
            ]
    
        title_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*FailureVsWarning Validation Operator Completed.*",
            },
        }
        divider_block = {
            "type": "divider"
        }

        query = {"blocks": [divider_block, title_block, divider_block]}

        status_element = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Status*: {}".format(status_text)},
        }
        query["blocks"].append(status_element)
        
        data_asset_identifiers_element = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Data Asset List:* {}".format(data_asset_identifiers)
            }
        }
        query["blocks"].append(data_asset_identifiers_element)
    
        if not success:
            failed_data_assets_element = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Failed Data Assets:* {}".format(failed_data_assets)
                }
            }
            query["blocks"].append(failed_data_assets_element)
    
        run_id_element = {
            "type": "section",
            "text":
                {
                    "type": "mrkdwn",
                    "text": "*Run ID:* {}".format(run_id),
                }
            ,
        }
        query["blocks"].append(run_id_element)
        
        timestamp_element = {
            "type": "section",
            "text":
                {
                    "type": "mrkdwn",
                    "text": "*Timestamp:* {}".format(timestamp),
                }
            ,
        }
        query["blocks"].append(timestamp_element)
        query["blocks"].append(divider_block)

        documentation_url = "https://docs.greatexpectations.io/en/latest/reference/validation_operators/warning_and_failure_expectation_suites_validation_operator.html"
        footer_section = {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Learn about FailureVsWarning Validation Operators at {}".format(documentation_url),
                }
            ],
        }
        query["blocks"].append(footer_section)
        
        return query

    def run(self, assets_to_validate, run_id):
        # NOTE : Abe 2019/09/12: We should consider typing this object, since it's passed between classes.
        # Maybe use a Store, since it's a key-value thing...?
        # For now, I'm NOT typing it until we gain more practical experience with operators and actions.
        return_obj = {
            "data_asset_identifiers": [],
            "success": None,
            "failure": {},
            "warning": {},
            "run_id": run_id
        }

        for item in assets_to_validate:
            batch = self._build_batch_from_item(item)

            # TODO : We should be using typed batch
            data_asset_identifier = DataAssetIdentifier(
                *self.data_context.normalize_data_asset_name(
                    batch._expectation_suite["data_asset_name"]
                )
            )
            run_id = run_id

            assert not data_asset_identifier is None
            assert not run_id is None

            return_obj["data_asset_identifiers"].append(data_asset_identifier)

            # NOTE : Abe 2019/09/12 : Perhaps this could be generalized to a loop.
            # I'm NOT doing that, because lots of user research suggests that these 3 specific behaviors
            # (failure, warning, quarantine) will cover most of the common use cases for
            # post-validation data treatment.

            failure_expectation_suite_identifier = ExpectationSuiteIdentifier(
                data_asset_name=data_asset_identifier,
                expectation_suite_name=self.expectation_suite_name_prefix + self.expectation_suite_name_suffixes[0]
            )

            failure_validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=failure_expectation_suite_identifier,
                run_id=run_id,
            )

            failure_expectation_suite = None
            try:
                failure_expectation_suite = self.data_context.stores[self.data_context.expectations_store_name].get(
                    failure_expectation_suite_identifier
                )

            # NOTE : Abe 2019/09/17 : I'm concerned that this may be too permissive, since
            # it will catch any error in the Store, not just KeyErrors. In the longer term, a better
            # solution will be to have the Stores catch other known errors and raise KeyErrors,
            # so that methods like this can catch and handle a single error type.
            except Exception as e:
                logger.debug("Failure expectation suite not found: {}".format(failure_expectation_suite_identifier))

            if failure_expectation_suite:
                return_obj["failure"][failure_validation_result_id] = {}
                failure_validation_result = batch.validate(failure_expectation_suite, result_format="SUMMARY")
                return_obj["failure"][failure_validation_result_id]["validation_result"] = failure_validation_result
                failure_actions_results = self._run_actions(
                    batch,
                    failure_expectation_suite_identifier,
                    failure_expectation_suite,
                    failure_validation_result,
                    run_id
                )
                return_obj["failure"][failure_validation_result_id]["actions_results"] = failure_actions_results

                if not failure_validation_result["success"] and self.stop_on_first_error:
                    break


            warning_expectation_suite_identifier = ExpectationSuiteIdentifier(
                data_asset_name=data_asset_identifier,
                expectation_suite_name=self.expectation_suite_name_prefix + self.expectation_suite_name_suffixes[1]
            )

            warning_validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=warning_expectation_suite_identifier,
                run_id=run_id,
            )

            warning_expectation_suite = None
            try:
                warning_expectation_suite = self.data_context.stores[self.data_context.expectations_store_name].get(
                    warning_expectation_suite_identifier
                )
            except Exception as e:
                logger.debug("Warning expectation suite not found: {}".format(warning_expectation_suite_identifier))

            if warning_expectation_suite:
                return_obj["warning"][warning_validation_result_id] = {}
                warning_validation_result = batch.validate(warning_expectation_suite, result_format="SUMMARY")
                return_obj["warning"][warning_validation_result_id]["validation_result"] = warning_validation_result
                warning_actions_results = self._run_actions(
                    batch,
                    warning_expectation_suite_identifier,
                    warning_expectation_suite,
                    warning_validation_result,
                    run_id
                )
                return_obj["warning"][warning_validation_result_id]["actions_results"] = warning_actions_results

        return_obj["success"] = all([val["validation_result"]["success"] for val in return_obj["failure"].values()])

        # NOTE: Eugene: 2019-09-24: Update the data doc sites?
        if self.slack_webhook:
            if self.notify_on == "all" or \
                    self.notify_on == "success" and return_obj["success"] or \
                    self.notify_on == "failure" and not return_obj["success"]:
                slack_query = self._build_slack_query(run_return_obj=return_obj)
                send_slack_notification(query=slack_query, slack_webhook=self.slack_webhook)

        return return_obj


