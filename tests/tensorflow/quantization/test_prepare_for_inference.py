"""
 Copyright (c) 2023 Intel Corporation
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

import pytest

from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_two_conv_test_model
from tests.tensorflow.quantization.utils import get_basic_quantization_config


@pytest.mark.parametrize("make_model_copy", (True, False))
def test_make_model_copy(make_model_copy):
    model = get_basic_two_conv_test_model()
    config = get_basic_quantization_config()
    config["target_device"] = "TRIAL"
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
    }
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    inference_model = compression_ctrl.prepare_for_inference(make_model_copy=make_model_copy)

    if make_model_copy:
        assert id(inference_model) != id(compression_model)
    else:
        assert id(inference_model) == id(compression_model)

    assert id(compression_model) == id(compression_ctrl.model)
