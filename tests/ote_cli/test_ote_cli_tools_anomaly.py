import importlib
import os
import shutil
import sys
from subprocess import run

import pytest

try:
    from helpers.dummy_dataset import TestDataset
except ImportError as e:
    raise e

from ote_cli.datasets.anomaly.dataset import JSONFromDataset
from ote_cli.registry import Registry

registry = Registry("external")


def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template["path"]))


def get_some_vars(template, root):
    template_dir = get_template_rel_dir(template)
    task_type = template["task_type"]
    work_dir = os.path.join(root, task_type)
    template_work_dir = os.path.join(work_dir, template_dir)
    algo_backend_dir = "/".join(template_dir.split("/")[:2])
    return template_dir, work_dir, template_work_dir, algo_backend_dir


def gen_parse_model_template_tests(task_type):
    class MyTests:
        pass

    root = "/tmp/ote_cli/"
    ote_dir = os.getcwd()

    test_id = 0
    for template in registry.filter(task_type=task_type).templates:

        @pytest.mark.run(order=test_id)
        @TestDataset()
        def test_ote_train(self, template=template, dataset_path="./datasets/MVTec", category="shapes"):
            with JSONFromDataset(dataset_path=os.path.join(dataset_path, category)) as json_path:
                template_dir, work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
                assert run(f"./{algo_backend_dir}/init_venv.sh {work_dir}/venv", check=True, shell=True).returncode == 0
                os.makedirs(template_work_dir, exist_ok=True)
                print(f"{template_work_dir=}")

                command_line = (
                    f"ote_train "
                    f'--train-ann-file {json_path/"train.json"} '
                    f"--train-data-roots {os.path.join(dataset_path,category)} "
                    f'--val-ann-file {json_path /"val.json"} '
                    f"--val-data-roots {os.path.join(dataset_path,category)} "
                    f"--save-weights {template_work_dir}/trained.pth "
                )
                print(
                    f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}"
                )
                assert (
                    run(
                        f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}",
                        check=True,
                        shell=True,
                    ).returncode
                    == 0
                )

        setattr(
            MyTests,
            "test_ote_train_" + template["task_type"] + "__" + get_template_rel_dir(template),
            test_ote_train,
        )
        test_id += 1

        @pytest.mark.run(order=test_id)
        @TestDataset()
        def test_ote_eval(self, template=template, dataset_path="./datasets/MVTec", category="shapes"):
            with JSONFromDataset(dataset_path=os.path.join(dataset_path, category)) as json_path:
                template_dir, work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
                assert run(f"./{algo_backend_dir}/init_venv.sh {work_dir}/venv", check=True, shell=True).returncode == 0
                os.makedirs(template_work_dir, exist_ok=True)
                print(f"{template_work_dir=}")

                command_line = (
                    f"ote_eval "
                    f'--test-ann-file {json_path/"test.json"} '
                    f"--test-data-roots {os.path.join(dataset_path,category)} "
                    f"--load-weights {template_work_dir}/trained.pth "
                )

                assert (
                    run(
                        f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}",
                        check=True,
                        shell=True,
                    ).returncode
                    == 0
                )

        setattr(
            MyTests, "test_ote_eval_" + template["task_type"] + "__" + get_template_rel_dir(template), test_ote_eval
        )
        test_id += 1

        @pytest.mark.run(order=test_id)
        @TestDataset()
        def test_ote_export(self, template=template, dataset_path="./datasets/MVTec", category="shapes"):
            template_dir, work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
            assert run(f"./{algo_backend_dir}/init_venv.sh {work_dir}/venv", check=True, shell=True).returncode == 0
            os.makedirs(template_work_dir, exist_ok=True)
            print(f"{template_work_dir=}")

            command_line = (
                f"ote_export "
                f"--labels shapes "
                f"--load-weights {template_work_dir}/trained.pth "
                f"--save-model-to {template_work_dir}/exported"
            )

            assert (
                run(
                    f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}",
                    check=True,
                    shell=True,
                ).returncode
                == 0
            )

        setattr(
            MyTests, "test_ote_export_" + template["task_type"] + "__" + get_template_rel_dir(template), test_ote_export
        )
        test_id += 1

    return MyTests


class TestOteCliWithANOMALY(gen_parse_model_template_tests(task_type="ANOMALY_CLASSIFICATION")):
    pass
