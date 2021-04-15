"""
 Copyright (c) 2021 Intel Corporation
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

import tensorflow as tf

from nncf.api.compression import CompressionLoss


class TFZeroCompressionLoss(CompressionLoss):
    def calculate(self, *args, **kwargs) -> Any:
        return tf.constant(0.)

    def statistics(self, quickly_collected_only: bool = False) -> Dict[str, object]:
        return {}
