import os
from pathlib import Path

import pytest
import torch

from nncf.torch.binarization.extensions import BinarizedFunctionsCPU
from nncf.torch.binarization.extensions import BinarizedFunctionsCUDA
from nncf.torch.extensions import NNCF_TIME_LIMIT_TO_LOAD_EXTENSION
from nncf.torch.extensions import ExtensionLoaderTimeoutException


def test_timeout_extension_loader(tmp_path, use_cuda):
    if not torch.cuda.is_available() and use_cuda is True:
        pytest.skip("Skipping CUDA test cases for CPU only setups")

    quant_func = BinarizedFunctionsCPU if use_cuda else BinarizedFunctionsCUDA

    os.environ[NNCF_TIME_LIMIT_TO_LOAD_EXTENSION] = "1"
    os.environ["TORCH_EXTENSIONS_DIR"] = tmp_path.as_posix()

    build_dir = Path(quant_func._loader.get_build_dir())
    lock_file = build_dir / "lock"
    lock_file.touch()
    with pytest.raises(ExtensionLoaderTimeoutException):
        quant_func.get("Quantize_forward")
