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

from ast import Tuple
from typing import List

import torch
from torch import nn

from nncf import NNCFConfig
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
from nncf.torch.graph.operator_metatypes import PTSumMetatype
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from tests.post_training.models import NNCFGraphToTest
from tests.post_training.models import NNCFGraphToTestDepthwiseConv
from tests.post_training.models import NNCFGraphToTestSumAggregation
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv


def get_single_conv_nncf_graph() -> NNCFGraphToTest:
    conv_layer_attrs = ConvolutionLayerAttributes(
                            weight_requires_grad=True,
                            in_channels=4, out_channels=4, kernel_size=(4, 4),
                            stride=1, groups=1, transpose=False,
                            padding_values=[])
    return NNCFGraphToTest(PTModuleConv2dMetatype, conv_layer_attrs, PTNNCFGraph)


def get_depthwise_conv_nncf_graph() -> NNCFGraphToTestDepthwiseConv:
    conv_layer_attrs = GroupNormLayerAttributes(False, 3, 3)
    return NNCFGraphToTestDepthwiseConv(PTDepthwiseConv2dSubtype, conv_layer_attrs)


def get_single_no_weight_matmul_nncf_graph() -> NNCFGraphToTest:
    return NNCFGraphToTest(PTModuleLinearMetatype, None, PTNNCFGraph)


def get_sum_aggregation_nncf_graph() -> NNCFGraphToTestSumAggregation:
    conv_layer_attrs = ConvolutionLayerAttributes(
                            weight_requires_grad=True,
                            in_channels=4, out_channels=4, kernel_size=(4, 4),
                            stride=1, groups=1, transpose=False,
                            padding_values=[])
    return NNCFGraphToTestSumAggregation(PTModuleConv2dMetatype,
                                         PTSumMetatype,
                                         conv_layer_attrs,
                                         PTNNCFGraph)


def get_nncf_network(model: torch.nn.Module,
                     input_shape: List[int] = [1, 3, 32, 32]):
    model.eval()
    nncf_config = NNCFConfig({
        'input_info': {
            'sample_size': input_shape.copy()
        }
    })
    nncf_network = create_nncf_network(
        model=model,
        config=nncf_config,
    )
    return nncf_network


def get_min_max_algo_for_test():
    params = PostTrainingQuantizationParameters()
    min_max_params = params.algorithms[MinMaxQuantization]
    params.algorithms = {MinMaxQuantization: min_max_params}
    return PostTrainingQuantization(params)


def mock_collect_statistics(mocker):
    _ = mocker.patch(
        'nncf.common.tensor_statistics.aggregator.StatisticsAggregator.collect_statistics', return_value=None)
    min_, max_ = 0., 1.
    min_, max_ = map(lambda x: torch.tensor(x), [min_, max_])
    _ = mocker.patch(
        'nncf.common.tensor_statistics.collectors.TensorStatisticCollectorBase.get_statistics',
        return_value=PTMinMaxTensorStatistic(min_, max_))
