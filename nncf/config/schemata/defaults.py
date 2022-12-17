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

TARGET_DEVICE = 'ANY'

NUM_BN_ADAPTATION_SAMPLES = 2000
NUM_INIT_SAMPLES = 256
MIN_PERCENTILE = 0.1
MAX_PERCENTILE = 99.9

PRECISION_INIT_BITWIDTHS = [2, 4, 8]
HAWQ_NUM_DATA_POINTS = 100
HAWQ_ITER_NUMBER = 200
HAWQ_TOLERANCE = 1e-4
HAWQ_COMPRESSION_RATIO = 1.5
HAWQ_DUMP_INIT_PRECISION_DATA = False

AUTOQ_ITER_NUMBER = 0
AUTOQ_COMPRESSION_RATIO = 0.15  # TODO (vshampor) do autoq and hawq follow the same convention here?
AUTOQ_EVAL_SUBSET_RATIO = 1.0
AUTOQ_WARMUP_ITER_NUMBER = 20

QUANTIZATION_PRESET = 'performance'
QUANTIZE_INPUTS = True
QUANTIZE_OUTPUTS = False
QUANTIZATION_EXPORT_TO_ONNX_STANDARD_OPS = False
QUANTIZATION_OVERFLOW_FIX = 'enable'
QUANTIZATION_BITS = 8
QUANTIZATION_PER_CHANNEL = False
QUANTIZATION_LOGARITHM_SCALE = False

ACTIVATIONS_QUANT_START_EPOCH = 1
WEIGHTS_QUANT_START_EPOCH = 1
LR_POLY_DURATION_EPOCHS = 30
STAGED_QUANTIZATION_BASE_LR = 1e-3
STAGED_QUANTIZATION_BASE_WD = 1e-5

BINARIZATION_MODE = 'xnor'

PRUNING_INIT = 0.0
PRUNING_SCHEDULE = 'exponential'
PRUNING_TARGET = 0.5
PRUNING_NUM_INIT_STEPS = 0
PRUNING_STEPS = 100
PRUNING_FILTER_IMPORTANCE = 'L2'
PRUNING_INTERLAYER_RANKING_TYPE = 'unweighted_ranking'
PRUNING_ALL_WEIGHTS = False
PRUNE_FIRST_CONV = False
PRUNE_BATCH_NORMS = True
PRUNE_DOWNSAMPLE_CONVS = False

PRUNING_LEGR_GENERATIONS = 400
PRUNING_LEGR_TRAIN_STEPS = 200
PRUNING_LEGR_MAX_PRUNING = 0.8
PRUNING_LEGR_RANDOM_SEED = 42

SPARSITY_INIT = 0.0
MAGNITUDE_SPARSITY_WEIGHT_IMPORTANCE = 'normed_abs'
SPARSITY_SCHEDULER = 'polynomial'
RB_SPARSITY_SCHEDULER = 'exponential'
SPARSITY_SCHEDULER_PATIENCE = 1
SPARSITY_SCHEDULER_POWER = 0.9
SPARSITY_SCHEDULER_CONCAVE = True
SPARSITY_TARGET = 0.5
SPARSITY_TARGET_EPOCH = 90
SPARSITY_FREEZE_EPOCH = 100
SPARSITY_SCHEDULER_UPDATE_PER_OPTIMIZER_STEP = False
SPARSITY_MULTISTEP_STEPS = [90]
SPARSITY_MULTISTEP_SPARSITY_LEVELS = [0.1, 0.5]
SPARSITY_LEVEL_SETTING_MODE = 'global'

KNOWLEDGE_DISTILLATION_SCALE = 1.0
KNOWLEDGE_DISTILLATION_TEMPERATURE = 1.0

AA_PATIENCE_EPOCHS = 3
AA_INITIAL_COMPRESSION_RATE_STEP = 0.1
AA_INITIAL_TRAINING_PHASE_EPOCHS = 5
AA_COMPRESSION_RATE_STEP_REDUCTION_FACTOR = 0.5
AA_LR_REDUCTION_FACTOR = 0.5
AA_MINIMAL_COMPRESSION_RATE_STEP = 0.025
AA_MAXIMAL_TOTAL_EPOCHS = 10000
