# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Type
import tempfile
from os import path as osp
import yaml

import numpy
import torch
import pytest
from mmdet.apis.ote.apis.detection.ote_utils import get_task_class
from mmdet.integration.nncf.utils import is_nncf_enabled

from ote_sdk.configuration.helper import create as ote_sdk_configuration_helper_create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelStatus,
)
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.entities.train_parameters import (
    UpdateProgressCallback, TrainParameters)

try:
    import hpopt
except ImportError as e:
    hpopt = None

from .e2e_test_system import DataCollector
from .training_tests_common import (
    KEEP_CONFIG_FIELD_VALUE,
    performance_to_score_name_value,
)

logger = logging.getLogger(__name__)


class BaseOTETestAction(ABC):
    _name: Optional[str] = None
    _with_validation = False
    _depends_stages_names: List[str] = []

    def __init__(*args, **kwargs):
        pass

    @property
    def name(self):
        return type(self)._name

    @property
    def with_validation(self):
        return type(self)._with_validation

    @property
    def depends_stages_names(self):
        return type(self)._depends_stages_names

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"name={self.name}, "
            f"with_validation={self.with_validation}, "
            f"depends_stages_names={self.depends_stages_names})"
        )

    def _check_result_prev_stages(self, results_prev_stages, list_required_stages):
        for stage_name in list_required_stages:
            if not results_prev_stages or stage_name not in results_prev_stages:
                raise RuntimeError(
                    f"The action {self.name} requires results of the stage {stage_name}, "
                    f"but they are absent"
                )

    @abstractmethod
    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        raise NotImplementedError("The main action method is not implemented")


class OTETestHPO(BaseOTETestAction):
    _name = "hpo"

    def __init__(
        self, dataset, labels_schema, template_path, num_training_iters,
        batch_size, auto_config):
        self.dataset = dataset
        self.labels_schema = labels_schema
        self.template_path = template_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size
        self.work_dir = tempfile.mkdtemp(prefix="ote-test-hpo-")
        self.auto_config = auto_config

        logger.debug(f"HPO work dir : {self.work_dir}")

        logger.info("hpopt configuration is loaded")
        with open(osp.dirname(self.template_path)
                  + "/hpo_config.yaml", 'r') as f:
            hpopt_cfg = yaml.safe_load(f)
        trainset_size = len(self.dataset.get_subset(Subset.TRAINING))
        valset_size = len(self.dataset.get_subset(Subset.VALIDATION))

        self.metric = hpopt_cfg['metric']
        hpopt_arguments = dict(
            search_alg='bayes_opt',
            search_space={'learning_parameters.learning_rate':hpopt.search_space(
                "quniform", [0.001, 0.01, 0.001])},
            save_path=self.work_dir,
            max_iterations=1,
            subset_ratio=1.0,
            early_stop = None,
            full_dataset_size=trainset_size,
            non_pure_train_ratio=trainset_size/(trainset_size+valset_size),
            num_init_trials=1,
            num_trials=2)

        if self.num_training_iters != KEEP_CONFIG_FIELD_VALUE:
            hpopt_arguments['num_full_iterations'] = int(self.num_training_iters)
        else:
            model_template = parse_model_template(self.template_path)
            params = ote_sdk_configuration_helper_create(
                model_template.hyper_parameters.data
            )
            hpopt_arguments['num_full_iterations'] = params.learning_parameters.num_iters

        if auto_config:
            hpopt_arguments['num_init_trials'] = 5
            search_sapce = dict()
            for key, val in hpopt_cfg['hp_space'].items():
                search_sapce[key] = hpopt.search_space(val['param_type'],
                                                       val['range'])
            hpopt_arguments['search_space'] = search_sapce
            del hpopt_arguments['num_trials']
            del hpopt_arguments['max_iterations']
            del hpopt_arguments['subset_ratio']

        self.hpo = hpopt.create(**hpopt_arguments)

    @staticmethod
    def _create_environment_and_task(params, labels_schema, model_template):
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )
        logger.info("Create base Task")
        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)
        return environment, task

    @staticmethod
    def set_hyperparameter(origin_hp, hp_config):
        for param_key, param_val in hp_config.items():
            param_key = param_key.split('.')

            target = origin_hp
            for val in param_key[:-1]:
                target = getattr(target, val)
            setattr(target, param_key[-1], param_val)

    def _run_ote_training(self, hp_config):
        class HpoDataset:
            def __init__(self, fullset, config=None, indices=None):
                self.fullset = fullset
                self.indices = indices
                self.subset_ratio = 1 if config is None else config['subset_ratio']

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, indx) -> dict:
                return self.fullset[self.indices[indx]]

            def __getattr__(self, name):
                if getattr(self.fullset, name, None) is not None:
                    if name == 'get_item_labels':
                        return self.__get_item_labels
                    if name == '_load_item':
                        return self.____load_item

                return getattr(self.fullset, name)

            def get_subset(self, subset: Subset):
                dataset = self.fullset.get_subset(subset)
                if subset != Subset.TRAINING or self.subset_ratio > 0.99:
                    return dataset

                indices = torch.randperm(len(dataset),
                                        generator=torch.Generator().manual_seed(42))
                indices = indices.tolist()
                indices = indices[:int(len(dataset)*self.subset_ratio)]

                return HpoDataset(dataset, config=None, indices=indices)

            # ClassificationDatasetAdapter
            def __get_item_labels(self, indx):
                return self.fullset.get_item_labels(self.indices[indx])

            def ____load_item(self, indx):
                return self.fullset._load_item(self.indices[indx])

        class HpoCallback(UpdateProgressCallback):
            def __init__(self, hp_config, metric, hpo_task):
                self.hp_config = hp_config
                self.metric = metric
                self.hpo_task = hpo_task

            def __call__(self, progress: float, score: Optional[float] = None):
                if score is not None and type(score) != list:
                    if isinstance(score, (int, float)):  # det
                        if hpopt.report(config=self.hp_config,
                                        score=score) == hpopt.Status.STOP:
                            self.hpo_task.cancel_training()
                    elif type(score) == numpy.float64:  # cls
                        if hpopt.report(config=self.hp_config,
                                        score=score.item()) == hpopt.Status.STOP:
                            self.hpo_task.cancel_training()

        dataset = HpoDataset(deepcopy(self.dataset), hp_config)

        model_template = parse_model_template(self.template_path)
        model_template.hpo = {'hp_config': hp_config, 'metric': self.metric}

        params = ote_sdk_configuration_helper_create(
            model_template.hyper_parameters.data
        )

        params.learning_parameters.num_iters = hp_config['iterations']

        if self.batch_size != KEEP_CONFIG_FIELD_VALUE:
            params.learning_parameters.batch_size = int(self.batch_size)

        self.set_hyperparameter(params, hp_config['params'])

        environment, task = self._create_environment_and_task(
            params, self.labels_schema, model_template
        )

        logger.debug(f"HPO trial {hp_config['trial_id']} Train model")
        output_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )

        train_param = TrainParameters(False, HpoCallback(
            hp_config, self.metric, task), None)

        task.train(dataset, output_model, train_param)

        hpopt.finalize_trial(hp_config)

    def _run_hpo(self, data_collector):
        logger.info('Begin HPO')
        while True:
            hp_config = self.hpo.get_next_sample()

            if hp_config is None:
                break

            logger.info(f"Begin HPO {hp_config['trial_id']} trial")
            self._run_ote_training(hp_config=hp_config)

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._run_hpo(data_collector)
        best_config = self.hpo.get_best_config()

        assert (
            hpopt.get_previous_status(self.work_dir) == hpopt.Status.COMPLETERESULT
        ), "HPO was failed"

        logger.info(f"End HPO, best_config : {best_config}")

        results = {"best_config" : best_config}
        return results


class OTETestTrainingAction(BaseOTETestAction):
    _name = "training"
    _depends_stages_names = ["hpo"] if hpopt is not None else []

    def __init__(
        self, dataset, labels_schema, template_path, num_training_iters, batch_size
    ):
        self.dataset = dataset
        self.labels_schema = labels_schema
        self.template_path = template_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

    @staticmethod
    def _create_environment_and_task(params, labels_schema, model_template):
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )
        logger.info("Create base Task")
        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)
        return environment, task

    def _get_training_performance_as_score_name_value(self):
        training_performance = getattr(self.output_model, "performance", None)
        if training_performance is None:
            raise RuntimeError("Cannot get training performance")
        return performance_to_score_name_value(training_performance)

    def _run_ote_training(self, data_collector, best_config=None):
        logger.debug(f"self.template_path = {self.template_path}")

        print(f"train dataset: {len(self.dataset.get_subset(Subset.TRAINING))} items")
        print(
            f"validation dataset: "
            f"{len(self.dataset.get_subset(Subset.VALIDATION))} items"
        )

        logger.debug("Load model template")
        self.model_template = parse_model_template(self.template_path)

        logger.debug("Set hyperparameters")
        params = ote_sdk_configuration_helper_create(
            self.model_template.hyper_parameters.data
        )
        if self.num_training_iters != KEEP_CONFIG_FIELD_VALUE:
            params.learning_parameters.num_iters = int(self.num_training_iters)
            logger.debug(
                f"Set params.learning_parameters.num_iters="
                f"{params.learning_parameters.num_iters}"
            )
        else:
            logger.debug(
                f"Keep params.learning_parameters.num_iters="
                f"{params.learning_parameters.num_iters}"
            )

        if self.batch_size != KEEP_CONFIG_FIELD_VALUE:
            params.learning_parameters.batch_size = int(self.batch_size)
            logger.debug(
                f"Set params.learning_parameters.batch_size="
                f"{params.learning_parameters.batch_size}"
            )
        else:
            logger.debug(
                f"Keep params.learning_parameters.batch_size="
                f"{params.learning_parameters.batch_size}"
            )

        if best_config:
            logger.debug(
                "Using best config from HPO, "
                f"set params.learning_parameters.learning_rate="
                f"{params.learning_parameters.batch_size}"
            )
            OTETestHPO.set_hyperparameter(params, best_config)

        logger.debug("Setup environment")
        self.environment, self.task = self._create_environment_and_task(
            params, self.labels_schema, self.model_template
        )

        logger.debug("Train model")
        self.output_model = ModelEntity(
            self.dataset,
            self.environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )

        self.copy_hyperparams = deepcopy(self.task._hyperparams)

        self.task.train(self.dataset, self.output_model)
        assert (
            self.output_model.model_status == ModelStatus.SUCCESS
        ), "Training was failed"

        score_name, score_value = self._get_training_performance_as_score_name_value()
        logger.info(f"performance={self.output_model.performance}")
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):

        if hpopt is None:
            best_config = None
        else:
            self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)
            best_config = results_prev_stages["hpo"]["best_config"]
        self._run_ote_training(data_collector, best_config)
        results = {
            "model_template": self.model_template,
            "task": self.task,
            "dataset": self.dataset,
            "environment": self.environment,
            "output_model": self.output_model,
        }
        return results


def run_evaluation(dataset, task, model):
    logger.debug("Evaluation: Get predictions on the dataset")
    predicted_dataset = task.infer(
        dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
    )
    resultset = ResultSetEntity(
        model=model,
        ground_truth_dataset=dataset,
        prediction_dataset=predicted_dataset,
    )
    logger.debug("Evaluation: Estimate quality on dataset")
    task.evaluate(resultset)
    evaluation_performance = resultset.performance
    logger.info(f"Evaluation: performance={evaluation_performance}")
    score_name, score_value = performance_to_score_name_value(evaluation_performance)
    return score_name, score_value


class OTETestTrainingEvaluationAction(BaseOTETestAction):
    _name = "training_evaluation"
    _with_validation = True
    _depends_stages_names = ["training"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_evaluation(self, data_collector, dataset, task, trained_model):
        logger.info("Begin evaluation of trained model")
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, task, trained_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info(
            f"End evaluation of trained model, results: {score_name}: {score_value}"
        )
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "dataset": results_prev_stages["training"]["dataset"],
            "task": results_prev_stages["training"]["task"],
            "trained_model": results_prev_stages["training"]["output_model"],
        }

        score_name, score_value = self._run_ote_evaluation(data_collector, **kwargs)
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


def run_export(environment, dataset, task, action_name, expected_optimization_type):
    logger.debug(
        f'For action "{action_name}": Copy environment for evaluation exported model'
    )

    environment_for_export = deepcopy(environment)

    logger.debug(f'For action "{action_name}": Create exported model')
    exported_model = ModelEntity(
        dataset,
        environment_for_export.get_model_configuration(),
        model_status=ModelStatus.NOT_READY,
    )
    logger.debug("Run export")
    task.export(ExportType.OPENVINO, exported_model)

    assert (
        exported_model.model_status == ModelStatus.SUCCESS
    ), f"In action '{action_name}': Export to OpenVINO was not successful"
    assert (
        exported_model.model_format == ModelFormat.OPENVINO
    ), f"In action '{action_name}': Wrong model format after export"
    assert (
        exported_model.optimization_type == expected_optimization_type
    ), f"In action '{action_name}': Wrong optimization type"

    logger.debug(
        f'For action "{action_name}": Set exported model into environment for export'
    )
    environment_for_export.model = exported_model
    return environment_for_export, exported_model


class OTETestExportAction(BaseOTETestAction):
    _name = "export"
    _depends_stages_names = ["training"]

    def _run_ote_export(self, data_collector, environment, dataset, task):
        self.environment_for_export, self.exported_model = run_export(
            environment,
            dataset,
            task,
            action_name=self.name,
            expected_optimization_type=ModelOptimizationType.MO,
        )

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "environment": results_prev_stages["training"]["environment"],
            "dataset": results_prev_stages["training"]["dataset"],
            "task": results_prev_stages["training"]["task"],
        }

        self._run_ote_export(data_collector, **kwargs)
        results = {
            "environment": self.environment_for_export,
            "exported_model": self.exported_model,
        }
        return results


def create_openvino_task(model_template, environment):
    logger.debug("Create OpenVINO Task")
    openvino_task_impl_path = model_template.entrypoints.openvino
    openvino_task_cls = get_task_class(openvino_task_impl_path)
    openvino_task = openvino_task_cls(environment)
    return openvino_task


class OTETestExportEvaluationAction(BaseOTETestAction):
    _name = "export_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "export", "training_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_export_evaluation(
        self,
        data_collector,
        model_template,
        dataset,
        environment_for_export,
        exported_model,
    ):
        logger.info("Begin evaluation of exported model")
        self.openvino_task = create_openvino_task(
            model_template, environment_for_export
        )
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, self.openvino_task, exported_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of exported model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "environment_for_export": results_prev_stages["export"]["environment"],
            "exported_model": results_prev_stages["export"]["exported_model"],
        }

        score_name, score_value = self._run_ote_export_evaluation(
            data_collector, **kwargs
        )
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


class OTETestPotAction(BaseOTETestAction):
    _name = "pot"
    _depends_stages_names = ["export"]

    def __init__(self, pot_subset=Subset.TRAINING):
        self.pot_subset = pot_subset

    def _run_ote_pot(
        self, data_collector, model_template, dataset, environment_for_export
    ):
        logger.debug("Creating environment and task for POT optimization")
        self.environment_for_pot = deepcopy(environment_for_export)
        self.openvino_task_pot = create_openvino_task(
            model_template, environment_for_export
        )

        self.optimized_model_pot = ModelEntity(
            dataset,
            self.environment_for_pot.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )
        logger.info("Run POT optimization")
        self.openvino_task_pot.optimize(
            OptimizationType.POT,
            dataset.get_subset(self.pot_subset),
            self.optimized_model_pot,
            OptimizationParameters(),
        )
        assert (
            self.optimized_model_pot.model_status == ModelStatus.SUCCESS
        ), "POT optimization was not successful"
        assert (
            self.optimized_model_pot.model_format == ModelFormat.OPENVINO
        ), "Wrong model format after pot"
        assert (
            self.optimized_model_pot.optimization_type == ModelOptimizationType.POT
        ), "Wrong optimization type"
        logger.info("POT optimization is finished")

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "environment_for_export": results_prev_stages["export"]["environment"],
        }

        self._run_ote_pot(data_collector, **kwargs)
        results = {
            "openvino_task_pot": self.openvino_task_pot,
            "optimized_model_pot": self.optimized_model_pot,
        }
        return results


class OTETestPotEvaluationAction(BaseOTETestAction):
    _name = "pot_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "pot", "export_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_pot_evaluation(
        self, data_collector, dataset, openvino_task_pot, optimized_model_pot
    ):
        logger.info("Begin evaluation of pot model")
        validation_dataset_pot = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset_pot, openvino_task_pot, optimized_model_pot
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of pot model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "dataset": results_prev_stages["training"]["dataset"],
            "openvino_task_pot": results_prev_stages["pot"]["openvino_task_pot"],
            "optimized_model_pot": results_prev_stages["pot"]["optimized_model_pot"],
        }

        score_name, score_value = self._run_ote_pot_evaluation(data_collector, **kwargs)
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


class OTETestNNCFAction(BaseOTETestAction):
    _name = "nncf"
    _depends_stages_names = ["training"]

    def _run_ote_nncf(
        self, data_collector, model_template, dataset, trained_model, environment
    ):
        logger.debug("Get predictions on the validation set for exported model")
        self.environment_for_nncf = deepcopy(environment)

        logger.info("Create NNCF Task")
        nncf_task_class_impl_path = model_template.entrypoints.nncf
        if not nncf_task_class_impl_path:
            pytest.skip("NNCF is not enabled for this template")

        if not is_nncf_enabled():
            pytest.skip("NNCF is not installed")

        logger.info("Creating NNCF task and structures")
        self.nncf_model = ModelEntity(
            dataset,
            self.environment_for_nncf.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )
        self.nncf_model.set_data("weights.pth", trained_model.get_data("weights.pth"))

        self.environment_for_nncf.model = self.nncf_model

        nncf_task_cls = get_task_class(nncf_task_class_impl_path)
        self.nncf_task = nncf_task_cls(task_environment=self.environment_for_nncf)

        logger.info("Run NNCF optimization")
        self.nncf_task.optimize(
            OptimizationType.NNCF, dataset, self.nncf_model, OptimizationParameters()
        )
        assert (
            self.nncf_model.model_status == ModelStatus.SUCCESS
        ), "NNCF optimization was not successful"
        assert (
            self.nncf_model.optimization_type == ModelOptimizationType.NNCF
        ), "Wrong optimization type"
        assert (
            self.nncf_model.model_format == ModelFormat.BASE_FRAMEWORK
        ), "Wrong model format"
        logger.info("NNCF optimization is finished")

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "trained_model": results_prev_stages["training"]["output_model"],
            "environment": results_prev_stages["training"]["environment"],
        }

        self._run_ote_nncf(data_collector, **kwargs)
        results = {
            "nncf_task": self.nncf_task,
            "nncf_model": self.nncf_model,
            "nncf_environment": self.environment_for_nncf,
        }
        return results


class OTETestNNCFEvaluationAction(BaseOTETestAction):
    _name = "nncf_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "nncf", "training_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_nncf_evaluation(self, data_collector, dataset, nncf_task, nncf_model):
        logger.info("Begin evaluation of nncf model")
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, nncf_task, nncf_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of nncf model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "dataset": results_prev_stages["training"]["dataset"],
            "nncf_task": results_prev_stages["nncf"]["nncf_task"],
            "nncf_model": results_prev_stages["nncf"]["nncf_model"],
        }

        score_name, score_value = self._run_ote_nncf_evaluation(
            data_collector, **kwargs
        )
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


class OTETestNNCFExportAction(BaseOTETestAction):
    _name = "nncf_export"
    _depends_stages_names = ["training", "nncf"]

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_nncf_export(
        self, data_collector, nncf_environment, dataset, nncf_task
    ):
        logger.info("Begin export of nncf model")
        self.environment_nncf_export, self.nncf_exported_model = run_export(
            nncf_environment,
            dataset,
            nncf_task,
            action_name=self.name,
            expected_optimization_type=ModelOptimizationType.NNCF,
        )
        logger.info("End export of nncf model")

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "nncf_environment": results_prev_stages["nncf"]["nncf_environment"],
            "dataset": results_prev_stages["training"]["dataset"],
            "nncf_task": results_prev_stages["nncf"]["nncf_task"],
        }

        self._run_ote_nncf_export(data_collector, **kwargs)
        results = {
            "environment": self.environment_nncf_export,
            "exported_model": self.nncf_exported_model,
        }
        return results


class OTETestNNCFExportEvaluationAction(BaseOTETestAction):
    _name = "nncf_export_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "nncf_export", "nncf_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_nncf_export_evaluation(
        self,
        data_collector,
        model_template,
        dataset,
        nncf_environment_for_export,
        nncf_exported_model,
    ):
        logger.info("Begin evaluation of NNCF exported model")
        self.openvino_task = create_openvino_task(
            model_template, nncf_environment_for_export
        )
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, self.openvino_task, nncf_exported_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of NNCF exported model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "nncf_environment_for_export": results_prev_stages["nncf_export"][
                "environment"
            ],
            "nncf_exported_model": results_prev_stages["nncf_export"]["exported_model"],
        }

        score_name, score_value = self._run_ote_nncf_export_evaluation(
            data_collector, **kwargs
        )
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


def get_default_test_action_classes() -> List[Type[BaseOTETestAction]]:
    res = [
        OTETestTrainingAction,
        OTETestTrainingEvaluationAction,
        OTETestExportAction,
        OTETestExportEvaluationAction,
        OTETestPotAction,
        OTETestPotEvaluationAction,
        OTETestNNCFAction,
        OTETestNNCFEvaluationAction,
        OTETestNNCFExportAction,
        OTETestNNCFExportEvaluationAction,
    ]
    if hpopt is not None:
        res.insert(0, OTETestHPO)
    else:
        logger.info("hpopt insn't installed. HPO test is skipped")

    return res