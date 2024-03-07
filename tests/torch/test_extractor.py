# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from torch import nn

import tests.post_training.test_templates.helpers as helpers
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch import wrap_model
from nncf.torch.extractor import extract_fused_subgraph_for_node
from nncf.torch.graph.transformations.commands import PTQuantizerInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.model_transformer import PTTransformationLayout
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QuantizationMode
from nncf.torch.quantization.layers import SymmetricQuantizer


@pytest.mark.parametrize(
    "model_cls, node_name",
    (
        (helpers.ConvBiasBNTestModel, "ConvBiasBNTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.ConvBNTestModel, "ConvBNTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.ConvTestModel, "ConvTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.CustomConvBNTestModel, "CustomConvBNTestModel/CustomConv[conv]/conv2d_0"),
        (helpers.CustomConvTestModel, "CustomConvTestModel/CustomConv[conv]/conv2d_0"),
    ),
)
def test_extract_fused_subgraph_for_node(model_cls, node_name):
    example_input = torch.ones(model_cls.INPUT_SIZE)

    model = wrap_model(model_cls().eval(), example_input=example_input, trace_parameters=True)
    graph = model.nncf.get_graph()
    node = graph.get_node_by_name(node_name)
    extracted_module = extract_fused_subgraph_for_node(node, model)

    with torch.no_grad():
        ret1 = model(example_input)
        ret2 = extracted_module(example_input)
        assert torch.any(torch.isclose(ret1, ret2))


@pytest.mark.parametrize(
    "model_cls, node_name",
    (
        (helpers.ConvBiasBNTestModel, "ConvBiasBNTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.ConvBNTestModel, "ConvBNTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.ConvTestModel, "ConvTestModel/Conv2d[conv]/conv2d_0"),
        (helpers.CustomConvBNTestModel, "CustomConvBNTestModel/CustomConv[conv]/conv2d_0"),
        (helpers.CustomConvTestModel, "CustomConvTestModel/CustomConv[conv]/conv2d_0"),
    ),
)
def test_extract_fused_subgraph_for_node_with_fq(model_cls, node_name):
    example_input = torch.ones(model_cls.INPUT_SIZE)

    model = wrap_model(model_cls().eval(), example_input=example_input, trace_parameters=True)

    transformer = PTModelTransformer(model)
    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        scale_shape=(1,),
        narrow_range=False,
        half_range=False,
        logarithm_scale=False,
    )

    fq = SymmetricQuantizer(qspec)
    command = PTQuantizerInsertionCommand(PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, node_name, input_port_id=1), fq)
    layout = PTTransformationLayout()
    layout.register(command)
    q_model = transformer.transform(layout)

    graph = q_model.nncf.get_graph()
    q_node = graph.get_node_by_name(node_name)
    extracted_module = extract_fused_subgraph_for_node(q_node, q_model)
    with torch.no_grad():
        ret1 = q_model(example_input)
        ret2 = extracted_module(example_input)
        assert torch.any(torch.isclose(ret1, ret2))

    if isinstance(extracted_module, nn.Sequential):
        assert extracted_module[0].w_fq is not None
    else:
        assert extracted_module.w_fq is not None
