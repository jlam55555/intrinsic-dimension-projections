"""
Offsets are the projection of a trainable weight matrix theta_d into a larger weight matrix theta_off (offset weights),
which are to be added to a non-trainable set of initial weights to be used as the weights for a projection layer.
"""
from typing import Tuple

import tensorflow as tf

class OffsetCreator():
    """Projects a lower-dimensional space (theta_d) to a higher-dimensional one (theta_D)"""

    def __init__(self,
                 size: int):
        """Initialize offset creator. Must specify the cardinality of the lower-dimensional (trainable) space"""
        self.size = size

    def create_offset(self,
                      shape: Tuple[int, ...]) -> tf.Variable:
        """Generate offset weights"""
        pass


class RandomDenseLinearOffsetCreator(OffsetCreator):
    """Random dense linear projection"""
    pass


class RFFOffsetCreator(OffsetCreator):
    """Random Fourier Features projection"""
    pass
