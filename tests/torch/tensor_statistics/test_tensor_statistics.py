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

from functools import partial
from typing import Dict, Tuple, Type

import pytest
import torch

from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanPercentileStatisticCollector
from nncf.common.tensor_statistics.collectors import MedianMADStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MixedMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import OfflineTensorStatisticCollector
from nncf.common.tensor_statistics.collectors import PercentileStatisticCollector
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.collectors import StatisticsNotCollectedError
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.torch.tensor import PTNNCFTensor


class TestCollectedStatistics:
    REF_INPUTS = [
        PTNNCFTensor(torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]])),
        PTNNCFTensor(torch.tensor([[4.5, 2.6, 3.7], [-1.3, -4, -3.5], [4.3, 5.8, 6.1]])),
    ]

    @pytest.mark.parametrize(
        ("collector", "reduction_shapes_vs_ref_statistic"),
        [
            (
                MinMaxStatisticCollector,
                {
                    ((1,), (0, 1)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([-4.0])), 
                        max_values=PTNNCFTensor(torch.tensor([6.1]))
                    ),
                    ((3, 1), (1,)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([[1.0], [-4.0], [4.0]])), max_values=PTNNCFTensor(torch.tensor([[4.5], [4.0], [6.1]]))
                    ),
                    ((1, 3), (0,)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([[-1.3, -4.0, -3.5]])), max_values=PTNNCFTensor(torch.tensor([[4.5, 5.8, 6.1]]))
                    ),
                    # Not supported for now:
                    # ((3, 3), ): MinMaxTensorStatistic(
                    #     min_values=torch.tensor([
                    #         [1.0, 2.0, 3.0],
                    #         [-1.3, -4, -3.5],
                    #         [4.0, 5.0, 6.0]
                    #     ]),
                    #     max_values=torch.tensor([
                    #         [4.5, 2.6, 3.7],
                    #         [1.3, 4.0, 3.5],
                    #         [4.3, 5.8, 6.1]
                    #     ]),
                    # ),
                },
            ),
            (
                partial(MeanMinMaxStatisticCollector, use_per_sample_stats=False),
                {
                    ((1,), (0, 1)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([-3.5])), max_values=PTNNCFTensor(torch.tensor([6.05]))
                    ),
                    ((3, 1), (1,)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([[1.8], [-3.5], [4.15]])),
                        max_values=PTNNCFTensor(torch.tensor([[3.75], [3.5], [6.05]])),
                    ),
                    ((1, 3), (0,)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([[-1.15, -3, -3.25]])), max_values=PTNNCFTensor(torch.tensor([[4.25, 5.4, 6.05]]))
                    ),
                },
            ),
            (
                partial(
                    MixedMinMaxStatisticCollector,
                    use_per_sample_stats=False,
                    use_means_of_mins=False,
                    use_means_of_maxs=True,
                ),
                {
                    ((1,), (0, 1)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([-4.0])), max_values=PTNNCFTensor(torch.tensor([6.05]))
                    ),
                    ((3, 1), (1,)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([[1.0], [-4.0], [4.0]])),
                        max_values=PTNNCFTensor(torch.tensor([[3.75], [3.5], [6.05]])),
                    ),
                    ((1, 3), (0,)): MinMaxTensorStatistic(
                        min_values=PTNNCFTensor(torch.tensor([[-1.3, -4.0, -3.5]])), max_values=PTNNCFTensor(torch.tensor([[4.25, 5.4, 6.05]]))
                    ),
                },
            ),
        ],
    )
    def test_collected_statistics_with_shape_convert(
        self,
        collector: Type[TensorStatisticCollectorBase],
        reduction_shapes_vs_ref_statistic: Dict[Tuple[ReductionAxes, ReductionAxes], TensorStatistic],
    ):
        for shapes in reduction_shapes_vs_ref_statistic.keys():
            output_shape, reduction_shape = shapes
            collector_obj = collector(use_abs_max=True, reduction_axes=reduction_shape)
            for input_ in TestCollectedStatistics.REF_INPUTS:
                collector_obj.register_input(input_)
            test_stats = collector_obj.get_statistics()
            assert reduction_shapes_vs_ref_statistic[shapes] == test_stats

    @pytest.mark.parametrize(
        ("collector", "reduction_shapes_vs_ref_statistic"),
        [
            (
                MedianMADStatisticCollector,
                {
                    (1,): MedianMADTensorStatistic(median_values=PTNNCFTensor(torch.tensor([2.8])), mad_values=PTNNCFTensor(torch.tensor([2.6]))),
                    (3, 1): MedianMADTensorStatistic(
                        median_values=PTNNCFTensor(torch.tensor([[2.8], [-2.5], [5.4]])),
                        mad_values=PTNNCFTensor(torch.tensor([[0.85], [1.1], [0.65]])),
                    ),
                    (1, 3): MedianMADTensorStatistic(
                        median_values=PTNNCFTensor(torch.tensor([[2.5, 2.3, 3.35]])), mad_values=PTNNCFTensor(torch.tensor([[1.9, 3.1, 2.7]]))
                    ),
                    # Not supported for now:
                    # (3, 3): MedianMADTensorStatistic(
                    #     median_values=torch.tensor([
                    #         [2.75, 2.3, 3.35],
                    #         [-1.15, -3, -3.25],
                    #         [4.15, 5.4, 6.05]
                    #     ]),
                    #     mad_values=torch.tensor([
                    #         [1.75, 0.3, 0.35],
                    #         [0.15, 1, 0.25],
                    #         [0.15, 0.4, 0.05]
                    #     ]),
                    # ),
                },
            ),
            (
                partial(PercentileStatisticCollector, percentiles_to_collect=[10.0]),
                {
                    (1,): PercentileTensorStatistic({10.0: PTNNCFTensor(torch.tensor([-3.15]))}),
                    (3, 1): PercentileTensorStatistic({10.0: PTNNCFTensor(torch.tensor([[1.5], [-3.75], [4.15]]))}),
                    (1, 3): PercentileTensorStatistic({10.0: PTNNCFTensor(torch.tensor([[-1.15, -3, -3.25]]))}),
                    # Not supported for now:
                    # (3, 3): PercentileTensorStatistic(
                    #     {
                    #         10.0: torch.tensor([
                    #             [1.35, 2.06, 3.07],
                    #             [-1.27, -3.8, -3.45],
                    #             [4.03, 5.08, 6.01]
                    #         ])
                    #     }
                    # ),
                },
            ),
            (
                partial(MeanPercentileStatisticCollector, percentiles_to_collect=[10.0]),
                {
                    (1,): PercentileTensorStatistic({10.0: PTNNCFTensor(torch.tensor([-2.9]))}),
                    (3, 1): PercentileTensorStatistic({10.0: PTNNCFTensor(torch.tensor([[2.0100], [-3.3500], [4.4000]]))}),
                    (1, 3): PercentileTensorStatistic({10.0: PTNNCFTensor(torch.tensor([[-0.3900, -1.9400, -1.9300]]))}),
                    # Not supported for now:
                    # (3, 3): PercentileTensorStatistic(
                    #     {
                    #         10.0: torch.tensor([
                    #             [ 2.7500,  2.3000,  3.3500],
                    #             [-1.1500, -3.0000, -3.2500],
                    #             [ 4.1500,  5.4000,  6.0500]
                    #         ])
                    #     }
                    # ),
                },
            ),
        ],
    )
    def test_collected_statistics(
        self,
        collector: Type[TensorStatisticCollectorBase],
        reduction_shapes_vs_ref_statistic: Dict[ReductionAxes, TensorStatistic],
    ):
        for shapes in reduction_shapes_vs_ref_statistic.keys():
            reduction_shape = shapes
            collector_obj = collector(reduction_axes=reduction_shape)
            for input_ in TestCollectedStatistics.REF_INPUTS:
                collector_obj.register_input(input_)
            test_stats = collector_obj.get_statistics()
            assert reduction_shapes_vs_ref_statistic[shapes] == test_stats

    COLLECTORS = [
        partial(MinMaxStatisticCollector, use_abs_max=False, output_shape=(1,)),
        partial(
            MixedMinMaxStatisticCollector,
            use_per_sample_stats=False,
            use_abs_max=False,
            use_means_of_mins=False,
            use_means_of_maxs=False,
            output_shape=(1,),
        ),
        partial(MeanMinMaxStatisticCollector, use_per_sample_stats=False, use_abs_max=False, output_shape=(1,)),
        MedianMADStatisticCollector,
        partial(PercentileStatisticCollector, percentiles_to_collect=[10.0]),
        partial(MeanPercentileStatisticCollector, percentiles_to_collect=[10.0]),
    ]

    @pytest.fixture(params=COLLECTORS)
    def collector_for_interface_test(self, request):
        collector_type = request.param
        return collector_type(reduction_shape=(1,))

    def test_collected_samples(self, collector_for_interface_test: TensorStatisticCollectorBase):
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        assert collector_for_interface_test.collected_samples() == len(TestCollectedStatistics.REF_INPUTS)

    def test_reset(self, collector_for_interface_test: TensorStatisticCollectorBase):
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        collector_for_interface_test.reset()
        assert collector_for_interface_test.collected_samples() == 0
        with pytest.raises(StatisticsNotCollectedError):
            collector_for_interface_test.get_statistics()

    def test_enable_disable(self, collector_for_interface_test: TensorStatisticCollectorBase):
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)

        collector_for_interface_test.disable()
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        assert collector_for_interface_test.collected_samples() == len(TestCollectedStatistics.REF_INPUTS)

        collector_for_interface_test.enable()
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        assert collector_for_interface_test.collected_samples() == 2 * len(TestCollectedStatistics.REF_INPUTS)

    OFFLINE_COLLECTORS = [
        partial(
            MixedMinMaxStatisticCollector,
            use_per_sample_stats=False,
            use_abs_max=False,
            use_means_of_mins=False,
            use_means_of_maxs=False,
            output_shape=(1,),
        ),
        partial(MeanMinMaxStatisticCollector, use_per_sample_stats=False, use_abs_max=False, output_shape=(1,)),
        MedianMADStatisticCollector,
        partial(PercentileStatisticCollector, percentiles_to_collect=[10.0]),
        partial(MeanPercentileStatisticCollector, percentiles_to_collect=[10.0]),
    ]

    REF_NUM_SAMPLES = 3

    @pytest.fixture(params=OFFLINE_COLLECTORS)
    def collector_for_num_samples_test(self, request):
        collector_type = request.param
        return collector_type(reduction_shape=(1,), num_samples=TestCollectedStatistics.REF_NUM_SAMPLES)

    def test_num_samples(self, collector_for_num_samples_test: OfflineTensorStatisticCollector):
        for input_ in TestCollectedStatistics.REF_INPUTS * 10:
            collector_for_num_samples_test.register_input(input_)
        assert collector_for_num_samples_test.collected_samples() == TestCollectedStatistics.REF_NUM_SAMPLES
