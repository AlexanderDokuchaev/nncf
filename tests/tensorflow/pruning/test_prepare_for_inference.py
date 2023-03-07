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
from tests.tensorflow.helpers import get_empty_config
from tests.tensorflow.pruning.helpers import get_basic_pruning_config
from tests.tensorflow.pruning.helpers import get_concat_test_model


@pytest.mark.parametrize("make_model_copy", (True, False))
@pytest.mark.parametrize("enable_quantization", (True, False), ids=("with_quantization", "no_quantization"))
def test_make_model_copy(make_model_copy, enable_quantization):
    input_shape = [1, 8, 8, 3]
    model = get_concat_test_model(input_shape)

    config = get_empty_config(input_sample_sizes=input_shape)
    config.update(
        {"compression": [{"algorithm": "filter_pruning", "pruning_init": 0.5, "params": {"prune_first_conv": True}}]}
    )
    if enable_quantization:
        config["compression"].append({"algorithm": "quantization", "preset": "mixed"})

    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    inference_model = compression_ctrl.prepare_for_inference(make_model_copy=make_model_copy)

    if make_model_copy:
        assert id(inference_model) != id(compression_model)
    else:
        assert id(inference_model) == id(compression_ctrl.model)

    if enable_quantization:
        for ctrl in compression_ctrl.child_ctrls:
            assert id(compression_ctrl.model) == id(ctrl.model)
