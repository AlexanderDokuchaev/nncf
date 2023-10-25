# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import functions as fns
from tests.shared.test_templates.template_test_nncf_tensor import TemplateTestNNCFTensorOperators


class TestNPNNCFTensorOperators(TemplateTestNNCFTensorOperators):
    @staticmethod
    def to_tensor(x):
        return np.array(x)

    def test_disable_error_handling(self):
        with fns.disable_error_handling(Tensor(self.to_tensor(1))):
            state = np.seterr()
            assert state["divide"] == "ignore"
            assert state["invalid"] == "ignore"
