"""
Copyright 2023 SambaNova Systems, Inc.

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

from generative_data_prep.utils import PackingConfig


def test_config_instantiation():
    """Basic test to verify that every Packing Config can be instantiated using from_str."""
    for choice in PackingConfig.get_choices():
        assert choice == PackingConfig.from_str(str(choice))
