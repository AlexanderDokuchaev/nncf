"""
 Copyright (c) 2019-2023 Intel Corporation
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
import copy

import numpy as np
import torch
from torch.quantization.fake_quantize import FakeQuantize

from nncf.api.compression import CompressionAlgorithmController
from nncf.torch.composite_compression import CompositeCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.filter_pruning.layers import FilterPruningMask
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.operations import ModelPruner
from nncf.torch.pruning.operations import PrunType
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer

SUPPORTED_ALGORITHMS = ["quantization", "filter_pruning"]


def prepare_for_inference(
    compressed_model: NNCFNetwork, compressed_ctrl: CompressionAlgorithmController, save_original_model: bool = False
) -> NNCFNetwork:
    """
    Prepare NNCFNetwork for inference
      - for quantization algorithm replace Replace NNCF quantizers modules to FakeQuantize.
      - for pruning_filter algorithm prune module by filling weights and bias with zeros.

    :param compressed_model: Compressed model.
    :param compressed_ctrl: Compression controller.
    :param save_original_model: `True` means that a copy of the model will be modified.

    :return NNCFNetwork: Converted model.
    """

    if not isinstance(compressed_model, NNCFNetwork):
        raise ValueError(f"Expected type of compressed_model is NNCFNetwork, but got {type(compressed_model)}.")

    if save_original_model:
        compressed_model = copy.deepcopy(compressed_model)

    compressed_model.train(False)

    if isinstance(compressed_ctrl, CompositeCompressionAlgorithmController):
        ctrls = compressed_ctrl.child_ctrls
    else:
        ctrls = [compressed_ctrl]

    # Check supported algorithms
    for controller in ctrls:
        if controller.name not in SUPPORTED_ALGORITHMS:
            raise RuntimeError(f"Function prepare_for_inference supports only {SUPPORTED_ALGORITHMS} algorithms.")

    # Strip the model
    for controller in ctrls:
        compressed_model = controller.strip_model(compressed_model)

    # Prepare the model for inference
    for controller in ctrls:
        if controller.name == "quantization":
            replace_quantizer_to_native_module(compressed_model)
        if controller.name == "filter_pruning":
            graph = compressed_model.get_original_graph()
            ModelPruner(compressed_model, graph, PT_PRUNING_OPERATOR_METATYPES, PrunType.FILL_ZEROS).prune_model()

    clean_operators(compressed_model)

    return compressed_model


def clean_operators(model: NNCFNetwork) -> None:
    """
    Remove all unused operators from the model.
    Conditions for removing operators:
      - disabled quantization operator;
      - all filer_pruning operators.

    :param model: Compressed model.
    """

    if hasattr(model, "external_quantizers"):
        for key in list(model.external_quantizers.keys()):
            op = model.external_quantizers[key]
            if isinstance(op, BaseQuantizer) and not op.is_enabled_quantization():
                model.external_quantizers.pop(key)

    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key)
                if isinstance(op.op, FilterPruningMask):
                    nncf_module.remove_pre_forward_operation(key)
                elif isinstance(op, BaseQuantizer) and not op.is_enabled_quantization():
                    nncf_module.remove_pre_forward_operation(key)

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.post_ops(key)
                if isinstance(op.op, FilterPruningMask):
                    nncf_module.remove_post_forward_operation(key)
                if isinstance(op, BaseQuantizer) and not op.is_enabled_quantization():
                    nncf_module.remove_post_forward_operation(key)


def replace_quantizer_to_native_module(model: NNCFNetwork) -> None:
    """
    Replace NNCF quantizer modules to PyTorch FakeQuantizer module.

    :param model: Target model.
    """

    for key in model.external_quantizers.keys():
        if model.external_quantizers[key].is_enabled_quantization():
            model.external_quantizers[key] = convert_to_fakequantizer(model.external_quantizers[key])

    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in nncf_module.pre_ops.keys():
                op = nncf_module.get_pre_op(key)
                if isinstance(op.op, BaseQuantizer) and op.op.is_enabled_quantization():
                    if op.op.is_half_range:
                        # Half range require to clamp weights of module
                        # Note: Half range used only for weight.
                        input_low, input_high = op.op.get_input_low_input_high()

                        data = nncf_module.weight.data
                        data = torch.min(torch.max(data, input_low), input_high)
                        data = op.op.quantize(data, execute_traced_op_as_identity=False)
                        nncf_module.weight.data = data

                    op.op = convert_to_fakequantizer(op.op)

        if hasattr(nncf_module, "post_ops"):
            for key in nncf_module.post_ops.keys():
                op = nncf_module.get_post_ops(key)
                if isinstance(op.op, BaseQuantizer):
                    if op.op.is_enabled_quantization():
                        op.op = convert_to_fakequantizer(op.op)


def convert_to_fakequantizer(nncf_quantizer: BaseQuantizer) -> FakeQuantize:
    """
    Convert BaseQuantizer module to FakeQuantize.

    :param quantizer: NNCF Quantizer module.

    :return: Instance of FakeQuantize similar to the input quantizer.
    """
    assert nncf_quantizer.num_bits == 8, "Support only 8bit quantization."

    # TODO: levels can be not corrected when change sign? It's not visible because set_level_ranges called in forward.
    nncf_quantizer.set_level_ranges()

    per_channel = nncf_quantizer.per_channel
    scale_shape = nncf_quantizer.scale_shape
    ch_axis = np.argmax(scale_shape)
    dtype = torch.qint8 if nncf_quantizer.level_low < 0 else torch.quint8

    if per_channel:
        observer = torch.ao.quantization.observer.PerChannelMinMaxObserver
    else:
        observer = torch.ao.quantization.observer.MinMaxObserver

    if isinstance(nncf_quantizer, SymmetricQuantizer):
        qscheme = torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
    elif isinstance(nncf_quantizer, AsymmetricQuantizer):
        qscheme = torch.per_channel_affine if per_channel else torch.per_tensor_affine

    quant_min, quant_max, scale, zero_point = nncf_quantizer.get_parameters_for_torch_fq()

    fakequantizer = FakeQuantize(
        observer=observer,
        quant_max=quant_max,
        quant_min=quant_min,
        dtype=dtype,
        qscheme=qscheme,
        eps=nncf_quantizer.eps,
    )

    if not per_channel:
        scale = scale.squeeze()
        zero_point = zero_point.squeeze()

    fakequantizer.scale = scale
    fakequantizer.ch_axis = ch_axis
    fakequantizer.zero_point = zero_point

    # Disable observer to save parameters
    fakequantizer.disable_observer()

    return fakequantizer
