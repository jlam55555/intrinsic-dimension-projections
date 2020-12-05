"""
Offsets are the projection of a trainable weight matrix theta_d into a larger weight matrix theta_off (offset weights),
which are to be added to a non-trainable set of initial weights to be used as the weights for a projection layer.
"""
from typing import Tuple

import tensorflow as tf

from keras_ext.intrinsic_weights import IntrinsicWeights


class OffsetCreator:
    """Projects a lower-dimensional space (theta_d) to a higher-dimensional one (theta_D)"""

    def create_offset(self, intrinsic_weights: IntrinsicWeights, output_shape: Tuple[int, ...]) -> tf.Variable:
        """Generate offset weights"""
        raise NotImplementedError('Please implement this method')


class RandomDenseLinearOffsetCreator(OffsetCreator):
    """Random dense linear projection"""

    def create_offset(self,
                      intrinsic_weights: IntrinsicWeights,
                      output_shape: Tuple[int, ...],
                      initializer: tf.initializers.Initializer = tf.initializers.random_normal,
                      dtype: tf.dtypes.DType = tf.keras.backend.floatx()) -> tf.Variable:
        # generate projection matrix (non-trainable)
        total_output_dim = 1
        for dim in output_shape:
            total_output_dim *= dim
        projection_matrix = tf.Variable(
            initial_value=initializer(shape=(intrinsic_weights.size, total_output_dim)),
            dtype=dtype,
            trainable=False,
            name=f'_projector')

        # generate and return offset matrix
        return intrinsic_weights.weights @ projection_matrix


class RFFOffsetCreator(OffsetCreator):
    """Random Fourier Features projection"""

    def create_offset(self, intrinsic_weights: IntrinsicWeights, output_shape: Tuple[int, ...]) -> tf.Variable:
        pass
