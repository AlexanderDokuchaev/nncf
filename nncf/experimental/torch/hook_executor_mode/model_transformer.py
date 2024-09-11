
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
from nncf.common.graph.model_transformer import ModelTransformer
from torch import nn

# from nncf.experimental.torch.hook_executor_mode.wrapper import nn.Module
from nncf.torch.graph.transformations.commands import PTInsertionCommand, PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_multidevice
import torch
from torch import nn

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_multidevice

from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.torch.hook_executor_mode import HookType
from nncf.experimental.torch.hook_executor_mode.wrapper import insert_hook
HOOK_TYPE_MAP = {
    TargetType.OPERATOR_PRE_HOOK: HookType.PRE_HOOK,
    TargetType.OPERATION_WITH_WEIGHTS: HookType.PRE_HOOK,
    TargetType.OPERATOR_POST_HOOK: HookType.POST_HOOK,
}

class FnModule(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class PTModelTransformer(ModelTransformer):
    def __init__(self, model: nn.Module):
        super().__init__(model)

        device = None
        if not is_multidevice(model):
            device = get_model_device(model)

        self._command_transformation_ordered_pairs = [
            (PTInsertionCommand, partial(self._apply_insertion_transformations, device=device)),
            (PTSharedFnInsertionCommand,partial(self._apply_shared_nodes_insertion, device=device) )
        ]

    def transform(self, transformation_layout: PTTransformationLayout) -> nn.Module:
        transformations = transformation_layout.transformations
        aggregated_transformations = defaultdict(list)
        for transformation in transformations:
            aggregated_transformations[transformation.__class__].append(transformation)

        model = self._model
        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)

        return model

    @staticmethod
    def _apply_insertion_transformations(
        model: nn.Module, transformations: List[PTInsertionCommand], device: Optional[torch.device]
    ) -> nn.Module:
        for transformation_command in transformations:
            target_point: PTTargetPoint = transformation_command.target_point
            print("----ADD: ", target_point)

            hook_type = HOOK_TYPE_MAP[target_point.target_type]
            print(hook_type)

            insertion_module = transformation_command.fn
            if not isinstance(insertion_module, nn.Module):
                insertion_module = FnModule(insertion_module)
            # model.hooks.insert_hook(
            insert_hook(
                model=model,
                hook_type=hook_type,
                group_name=transformation_command.hooks_group_name,
                op_name=target_point.target_node_name.replace(".", ":"),
                port_id=target_point.input_port_id or 0,
                module=insertion_module,
            )


        return model

    @staticmethod
    def _apply_shared_nodes_insertion(
        model: nn.Module,
        transformations: List[PTSharedFnInsertionCommand],
        device: Optional[torch.device],
    ) -> nn.Module:

        compression_type_vs_transformations = defaultdict(list)
        for transformation in transformations:
            compression_type_vs_transformations[transformation.compression_module_type].append(transformation)

        for compression_module_type, commands in compression_type_vs_transformations.items():
            for command in commands:
                for target_point in command.target_points:
                    hook_type = HOOK_TYPE_MAP[target_point.target_type]

                    #model.hooks.insert_hook(
                    insert_hook(
                        model=model,
                        hook_type=hook_type,
                        group_name=command.hooks_group_name,
                        op_name=target_point.target_node_name.replace(".", ":"),
                        port_id=target_point.input_port_id or 0,
                        module=command.fn,
                    )
        return model

    # @staticmethod
    # def _apply_shared_node_insertion_with_compression_type(
    #     model: nn.Module,
    #     transformations: List[PTSharedFnInsertionCommand],
    #     device: Optional[torch.device],
    #     compression_module_type: ExtraCompressionModuleType,
    # ):
    #     """
    #     Does _apply_shared_nodes_insertion with specified compression model type which will be
    #     used for each transformation command.

    #     :param model: Model to apply transformations.
    #     :param transformations: List of the bias correction transformations.
    #     :param device: Target device for the insertion functions. Applies only to
    #         functions which are subclassed from torch.nn.Module. Do nothing in case device is None.
    #     :param compression_module_type: Common compression module type for all commands.
    #     :return: A modified NNCFNetwork.
    #     """
    #     if not model.nncf.is_compression_module_registered(compression_module_type):
    #         model.nncf.register_compression_module_type(compression_module_type)

    #     insertion_commands: List[PTInsertionCommand] = []

    #     for shared_command in transformations:
    #         fn = shared_command.fn
    #         if device is not None:
    #             fn.to(device)

    #         model.nncf.add_compression_module(shared_command.op_name, fn, compression_module_type)

    #         for target_point in shared_command.target_points:
    #             fn = ExternalOpCallHook(
    #                 compression_module_type_to_attr_name(compression_module_type), shared_command.op_name
    #             )
    #             insertion_commands.append(
    #                 PTInsertionCommand(
    #                     target_point,
    #                     fn,
    #                     priority=shared_command.priority,
    #                     hooks_group_name=shared_command.hooks_group_name,
    #                 )
    #             )

    #     return PTModelTransformer._apply_insertion_transformations(model, insertion_commands, device)
