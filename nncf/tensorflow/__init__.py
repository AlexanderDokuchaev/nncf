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
import tensorflow
from pkg_resources import parse_version

from nncf import nncf_logger
from nncf.common.logging.logger import warn_bkc_version_mismatch
# pylint: skip-file
from nncf.version import BKC_TF_VERSION

tensorflow_version = parse_version(tensorflow.__version__).base_version
if not tensorflow_version.startswith(BKC_TF_VERSION[:-2]):
    warn_bkc_version_mismatch("tensorflow", BKC_TF_VERSION, tensorflow.__version__)
elif not ("2.4" <= tensorflow_version[:3] <= "2.8"):
    raise RuntimeError(
        f"NNCF only supports 2.4.0 <= tensorflow <= 2.8.*, while current tensorflow version is {tensorflow.__version__}"
    )


from nncf.common.accuracy_aware_training.training_loop import AdaptiveCompressionTrainingLoop
from nncf.common.accuracy_aware_training.training_loop import EarlyExitCompressionTrainingLoop
from nncf.tensorflow.helpers import create_compressed_model
from nncf.tensorflow.helpers.callback_creation import create_compression_callbacks
from nncf.tensorflow.initialization import register_default_init_args
from nncf.tensorflow.pruning.filter_pruning import algorithm as filter_pruning_algorithm
# Required for correct COMPRESSION_ALGORITHMS registry functioning
from nncf.tensorflow.quantization import algorithm as quantization_algorithm
from nncf.tensorflow.sparsity.magnitude import algorithm as magnitude_sparsity_algorithm
from nncf.tensorflow.sparsity.rb import algorithm as rb_sparsity_algorithm
