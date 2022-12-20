"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Any
from typing import Dict
from typing import Tuple
from typing import TypeVar

from nncf import NNCFConfig
from nncf.common.compression import BaseCompressionAlgorithmBuilder
from nncf.common.graph import NNCFGraph
from nncf.common.scopes import check_scopes_in_graph
from nncf.tensorflow.graph.converter import TFModelConverter
from nncf.tensorflow.graph.converter import TFModelConverterFactory
from nncf.tensorflow.graph.model_transformer import TFModelTransformer

TModel = TypeVar('TModel')


class TFCompressionAlgorithmBuilder(BaseCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable algorithm-specific compression during fine-tuning.
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        compression_lr_multiplier = \
            config.get_redefinable_global_param_value_for_algo('compression_lr_multiplier', self.name)
        if compression_lr_multiplier is not None:
            raise Exception('compression_lr_multiplier is not supported when your work with a TF model in NNCF. '
                            'Please remove the compression_lr_multiplier attribute from your NNCFConfig.')

    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Implementation of get_state that returns state without builder name.

        :return: Returns a dictionary with Python data structures
            (dict, list, tuple, str, int, float, True, False, None) that represents state of the object.
        """
        return {}

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Implementation of load state that takes state without builder name.

        :param state_without_name: Output of `_get_state_without_name()` method.
        """

    def apply_to(self, model: TModel) -> TModel:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """
        transformation_layout = self.get_transformation_layout(model)
        transformer = TFModelTransformer(model)
        transformed_model = transformer.transform(transformation_layout)

        if self.should_init:
            self.initialize(transformed_model)

        return transformed_model

    def _get_model_converter_and_graph(self, model: TModel) -> Tuple[TFModelConverter, NNCFGraph]:
        """
        Check ignored/target scopes before return model converter and model graph.

        :param model: The original uncompressed model.

        :return convertor: Converter for TF models.
        :return nncf_graph: The `NNCFGraph` object that represents the TF model.
        """
        converter = TFModelConverterFactory.create(model)
        nncf_graph = converter.convert()

        check_scopes_in_graph(nncf_graph, self.ignored_scopes, self.target_scopes)

        return converter, nncf_graph
