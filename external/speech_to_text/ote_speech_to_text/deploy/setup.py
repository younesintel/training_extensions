"""
 Copyright (c) 2020-2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIR = Path(__file__).resolve().parent

with open(SETUP_DIR / "requirements.txt", "r", encoding="utf8") as f:
    required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
package_dir = {packages[0]: str(SETUP_DIR / packages[0])}

setup(
    name=packages[0],
    version="0.0",
    author="Intel® Corporation",
    license="Copyright (c) Intel - All Rights Reserved. "
    "Unauthorized copying of any part of the software via any medium is strictly prohibited. "
    "Proprietary and confidential.",
    description="Demo based on ModelAPI classes",
    packages=packages,
    package_dir=package_dir,
    package_data={
        packages[0]: ["*.json", "data/*"],
    },
    install_requires=required
)
