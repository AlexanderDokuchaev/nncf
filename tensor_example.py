from typing import List
import torch
from torch import nn



###########################################
# Minimal implementation of a custom tensor subclass
# https://docs.pytorch.org/ao/stable/subclass_basic.html#quantization-with-tensor-subclasses
class Int8SymmetricTensor(torch.Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: torch.Tensor, scale: torch.Tensor):

        return torch.Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            strides=int_data.stride(),
            storage_offset=int_data.storage_offset(),
            dtype=scale.dtype,
            device=int_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: torch.Tensor, scale: torch.Tensor):
        self.int_data = int_data
        self.scale = scale

    def __repr__(self):
        return f"Int8SymmetricTensor(int_data={repr(self.int_data)}, scale={repr(self.scale)})"

    def __tensor_flatten__(self):
        return ["int_data", "scale"], None

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, extra_metadata, outer_size=None, outer_stride=None):
        assert extra_metadata is None
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        return Int8SymmetricTensor(int_data, scale)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        print(f"__torch_dispatch__ called: {func.__name__}")
        if kwargs is None:
            kwargs = {}
        if func not in op_implementations_dict:
            raise AssertionError(
                f"Int8SymmetricTensor does not yet support op: {str(func)}"
            )
        return op_implementations_dict[func](func, *args, **kwargs)


####################################################
## Mapping own implementation
####################################################

op_implementations_dict = {}

def register_op(ops: List[torch._ops.OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations_dict
        for op in ops:
            op_implementations_dict[op] = op_impl
        return op_impl

    return impl_decorator



from torch.utils._python_dispatch import return_and_correct_aliasing


# matmul impl
@register_op([torch.ops.aten.mm.default])
def int8_mm(func, x, weight):
    print("int8_mm called")
    assert isinstance(weight, Int8SymmetricTensor), (
        "Int8SymmetricTensor: matmul currently only supports the weight in low precision, not the input!"
    )
    return torch.mm(x, weight.int_data.to(x.dtype)) * weight.scale


# implementation of most view operations
@register_op(
    [
        torch.ops.aten.detach.default,
        torch.ops.aten.t.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    ]
)
def int8_view_ops(func, *args, **kwargs):
    print(f"int8_view_ops called, {func.__name__}")

    assert isinstance(args[0], Int8SymmetricTensor)
    out_data = func(args[0].int_data, *args[1:], **kwargs)
    out_scale = func(args[0].scale, *args[1:], **kwargs)
    out = Int8SymmetricTensor(out_data, out_scale)
    # "return_and_correct_aliasing" here is needed for torch.compile support.
    # It effectively tells the compiler that the output of this view op aliases its input.
    # At some point, we're hoping to infer this automatically and kill this extra API!
    return return_and_correct_aliasing(func, args, kwargs, out)

@register_op([torch.ops.aten.copy_.default])
def int8_copy(func, *args, **kwargs):
    return Int8SymmetricTensor(args[0].int_data.clone(), args[0].scale.clone())



###################################################
# Usage
###################################################


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 16, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


float_model = ToyModel()
exmaple_input = torch.rand(1, 8, dtype=torch.float32)
output = float_model(exmaple_input)


# 1. Replace to custom weight

print("\nquantized_model: creation")
quantized_model = ToyModel()
q_tensor = Int8SymmetricTensor(torch.randint(-128, 128, (16, 8), dtype=torch.int8), torch.rand(16, 1, dtype=torch.float32))
quantized_model.linear.weight = nn.Parameter(q_tensor, requires_grad=False)

print("\nquantized_model: infer")
q_output = quantized_model(exmaple_input)


# 2. Save / load

state = quantized_model.state_dict()
print(state)

torch.save(state, "quantized_model.pth")
torch.serialization.add_safe_globals([Int8SymmetricTensor])
torch.serialization.safe_globals([Int8SymmetricTensor])
loaded_state = torch.load("quantized_model.pth")
print(loaded_state)

quantized_model.load_state_dict(loaded_state)


# 3. NNCF wrapper

# All function under torch_dispatch is hiden from NNCF, so we need to wrap the model with NNCF to get the graph.
# Possible add node attibute tensor_class/type, compressed or not or something else
from nncf.torch.model_creation import wrap_model
from nncf.torch.function_hook import register_post_function_hook
wrapped = wrap_model(quantized_model, exmaple_input, trace_parameters=True)
graph = wrapped.get_graph()
for n in graph.get_all_nodes():
    print("node", n)
for e in graph.get_all_edges():
    print("edge", e)



# 4. Insertion hooks on quantized weight

# Hook triggered before torch_dispatch and got Int8SymmetricTensor on input
class Hook(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("Hook called:", x)
        return x

register_post_function_hook(wrapped.model, "linear.weight", 0, Hook())

wrapped.model(exmaple_input)



# 5. convert to OpenVINO

# In OV tensor converted by tesnor.numpy() in result raised error
# overriding numpy() only by decompression of tensor but in OV node it will be float32 tensor. not int8+decompress
# Possible sollutions:
# 1. Strip model by int8 tesnor and decompression post_hook (as it works now)
# 2. Provide some API to convert Int8SymmetricTensor to subgraph in OV (requared change in OV, and method to convert to Int8SymmetricTensor to subgraph on NNCF side)


import openvino as ov

ov_model = ov.convert_model(wrapped.model, example_input=exmaple_input)
