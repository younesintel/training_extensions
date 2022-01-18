# pylint: skip-file

import logging
import os
import pickle

from anomaly_classification.debug import load_dataset
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity

logger = get_logger(__name__)


if __name__ == "__main__":
    import argparse

    from ote_sdk.entities.model import ModelEntity, ModelStatus
    from ote_sdk.entities.resultset import ResultSetEntity

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("dump_path")
        return parser.parse_args()

    def main():
        args = parse_args()
        assert os.path.exists(args.dump_path)

        output_model = None
        train_dataset = None

        with open(args.dump_path, "rb") as f:
            while True:
                print("reading dump record...")
                _logger = logging.getLogger()
                _logger.setLevel(logging.ERROR)
                try:
                    dump = pickle.load(f)
                except EOFError:
                    print("no more records found in the dump file")
                    break
                _logger.setLevel(logging.INFO)

                task = dump["task"]
                # Disable debug dump when replay another debug dump
                task.task_environment.get_hyper_parameters().debug_parameters.enable_debug_dump = False
                method_args = {}

                entrypoint = dump["entrypoint"]
                print("*" * 80)

                print(f"{type(task)=}, {entrypoint=}")
                print("=" * 80)

                while True:
                    action = input("[r]eplay, [s]kip or [q]uit : [r] ")
                    action = action.lower()
                    if action == "":
                        action = "r"
                    if action not in {"r", "s", "q"}:
                        continue
                    else:
                        break

                if action == "s":
                    print("skipping the step replay")
                    continue
                if action == "q":
                    print("quiting dump replay session")
                    exit(0)

                print("replaying the step")

                if entrypoint == "train":
                    method_args["dataset"] = load_dataset(dump["arguments"]["dataset"])
                    train_dataset = method_args["dataset"]
                    method_args["output_model"] = ModelEntity(
                        method_args["dataset"], task.task_environment, model_status=ModelStatus.NOT_READY
                    )
                    output_model = method_args["output_model"]
                    method_args["train_parameters"] = None
                elif entrypoint == "infer":
                    method_args["dataset"] = load_dataset(dump["arguments"]["dataset"])
                    method_args["inference_parameters"] = None
                elif entrypoint == "export":
                    method_args["output_model"] = ModelEntity(
                        train_dataset, task.task_environment, model_status=ModelStatus.NOT_READY
                    )
                    output_model = method_args["output_model"]
                    method_args["export_type"] = dump["arguments"]["export_type"]
                elif entrypoint == "evaluate":
                    output_model = ModelEntity(DatasetEntity(), task.task_environment, model_status=ModelStatus.SUCCESS)
                    output_model.configuration.label_schema = task.task_environment.label_schema
                    method_args["output_result_set"] = ResultSetEntity(
                        model=output_model,
                        ground_truth_dataset=load_dataset(
                            dump["arguments"]["output_resultset"]["ground_truth_dataset"]
                        ),
                        prediction_dataset=load_dataset(dump["arguments"]["output_resultset"]["prediction_dataset"]),
                    )
                    method_args["evaluation_metric"] = dump["arguments"]["evaluation_metric"]
                else:
                    raise RuntimeError(f"Unknown {entrypoint=}")

                output = getattr(task, entrypoint)(**method_args)
                print(f"\nOutput {type(output)=}\n\n\n\n")

    main()
