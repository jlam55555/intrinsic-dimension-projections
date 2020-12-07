"""
Offsets are the projection of a trainable weight matrix theta_d into a larger weight matrix theta_off (offset weights),
which are to be added to a non-trainable set of initial weights to be used as the weights for a projection layer.
"""
import tensorflow as tf

from keras_ext.intrinsic_weights import IntrinsicWeights


class OffsetCreator:
    """Projects a lower-dimensional space (theta_d) to a higher-dimensional one (theta_D)"""

    def create_offset(self,
                      intrinsic_weights: IntrinsicWeights,
                      output_shape: tf.TensorShape,
                      initializer: tf.initializers.Initializer = tf.initializers.random_normal,
                      dtype: tf.dtypes.DType = tf.keras.backend.floatx(),
                      name: str = None) -> tf.function:
        """Generate offset weights"""
        raise NotImplementedError('Please implement this method')

    def create_weight(self,
                      intrinsic_weights: IntrinsicWeights,
                      output_shape: tf.TensorShape,
                      initializer: tf.initializers.Initializer = tf.initializers.random_normal(),
                      dtype: tf.dtypes.DType = tf.keras.backend.floatx(),
                      name: str = None) -> tf.function:
        """Create initial weights and offset, returns weights (sum of the two) as a callable graph"""
        initial_weights_initializer = tf.initializers.RandomNormal()
        initial_weights = tf.Variable(initial_weights_initializer(shape=output_shape), trainable=False)

        offset_graph = self.create_offset(intrinsic_weights,
                                          output_shape,
                                          initializer,
                                          dtype,
                                          name)

        # @tf.function
        # def weight_graph():
        return initial_weights + offset_graph

        # return initial_weights, weight_graph


class RandomDenseLinearOffsetCreator(OffsetCreator):
    """Random dense linear projection"""

    def create_offset(self,
                      intrinsic_weights: IntrinsicWeights,
                      output_shape: tf.TensorShape,
                      initializer: tf.initializers.Initializer = tf.initializers.RandomNormal(),
                      dtype: tf.dtypes.DType = tf.keras.backend.floatx(),
                      name: str = None) -> tf.function:
        initializer = tf.initializers.get(initializer)

        # generate random projection matrix (non-trainable)
        total_output_dim = 1
        for dim in output_shape:
            total_output_dim *= dim
        projection_matrix = tf.Variable(initial_value=initializer(shape=(intrinsic_weights.size, total_output_dim)),
                                        dtype=dtype,
                                        trainable=False,
                                        name=f'{name}_projector')

        # generate and return offset matrix (only intrinsic weights are trainable)
        # @tf.function
        # def offset_graph():
        return tf.reshape(tf.matmul(intrinsic_weights.weights, projection_matrix, name=f'{name}_offset_unshaped'),
                          shape=output_shape,
                          name=f'{name}_offset')

        # return offset_graph


class RFFOffsetCreator(OffsetCreator):
    """Random Fourier Features projection"""
    pass
