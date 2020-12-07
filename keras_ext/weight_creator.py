"""
Generates weights given a desired output shape and intrinsic weights. How the projection actually works is up to the
subclass implementation.

Notation:
theta_D = theta_0 + f(theta_d)
- theta_D := output weights
- theta_0 := initial weights (non-trainable)
- f := projection method (non-trainable)
- theta_d := intrinsic weights (trainable)
- f(theta_d) := "offset", theta_off

To make theta_d trainable (to make it part of the model graph), return a function to be called in a layer's call()
method so that the operation is not immediately evaluated.

TODO: things to experiment with:
    - implement sparse/fastfood as well? (only improves performance, not compressibility)
    - same intrinsic weights for all layers in model?
    - trainable projection method? How can we do this while keeping parameterization small?
"""
import tensorflow as tf

from keras_ext.intrinsic_weights import IntrinsicWeights


class WeightCreator:
    """Generate weights given a desired output shape and an intrinsic matrix"""

    def __init__(self,
                 initial_weight_initializer: tf.initializers.Initializer = tf.initializers.RandomNormal()):
        self.initial_weight_initializer = initial_weight_initializer

    def _create_weight_offset(self,
                              output_shape: tf.TensorShape,
                              intrinsic_weights: IntrinsicWeights,
                              name: str = None,
                              dtype: tf.dtypes.DType = tf.keras.backend.floatx()) -> tf.function:
        """Create weight projection f(theta_d), return thunk that calculates this when called"""
        raise NotImplementedError

    def create_weight(self,
                      output_shape: tf.TensorShape,
                      intrinsic_weights: IntrinsicWeights,
                      name: str = None,
                      dtype: tf.dtypes.DType = tf.keras.backend.floatx()) -> tf.function:
        """Create weight theta_D, return thunk that calculates theta_D when called"""

        # create initial weight
        theta_0 = tf.Variable(initial_value=self.initial_weight_initializer(shape=output_shape),
                              dtype=dtype,
                              trainable=False,
                              name=f'{name}_theta_0')

        # create offset
        theta_off_thunk = self._create_weight_offset(output_shape, intrinsic_weights, name, dtype)

        # return thunk that calculates theta_D
        return tf.function(lambda: tf.add(theta_0, theta_off_thunk(), name=f'{name}_weight'))


class DenseLinearWeightCreator(WeightCreator):
    """Generates weights using a dense random linear projection"""

    def __init__(self,
                 initial_weight_initializer: tf.initializers.Initializer = tf.initializers.RandomNormal(),
                 projection_matrix_initializer: tf.initializers.Initializer = tf.initializers.RandomNormal()):
        super().__init__(initial_weight_initializer)
        self.initial_weight_initializer = tf.initializers.get(initial_weight_initializer)
        self.projection_matrix_initializer = tf.initializers.get(projection_matrix_initializer)

    def _create_weight_offset(self,
                              output_shape: tf.TensorShape,
                              intrinsic_weights: IntrinsicWeights,
                              name: str = None,
                              dtype: tf.dtypes.DType = tf.keras.backend.floatx()) -> tf.function:
        """Create dense weight projection"""

        # create random projection matrix
        total_output_dim = 1
        for dim in output_shape:
            total_output_dim *= dim
        projection_matrix = tf.Variable(self.projection_matrix_initializer(shape=(intrinsic_weights.size,
                                                                                  total_output_dim)),
                                        dtype=dtype,
                                        trainable=False,
                                        name=f'{name}_projector')

        # return thunk that calculates the projection when called
        return tf.function(lambda: tf.reshape(intrinsic_weights @ projection_matrix,
                                              shape=output_shape,
                                              name=f'{name}_dense_offset'))


class RFFWeightCreator(WeightCreator):
    """Create random fourier feature projection -- first "frequency-samples" intrinsic dimension vector, then projects
    that onto full weight space"""

    def __init__(self,
                 initial_weight_initializer: tf.initializers.Initializer = 'glorot_uniform',
                 projection_matrix_initializer: tf.initializers.Initializer = 'glorot_uniform',
                 frequency_samples: int = 50,
                 frequency_sample_mean: float = 0,
                 frequency_sample_std: float = 1):
        super().__init__(initial_weight_initializer)
        self.initial_weight_initializer = tf.initializers.get(initial_weight_initializer)
        self.projection_matrix_initializer = tf.initializers.get(projection_matrix_initializer)
        self.frequency_samples = frequency_samples
        self.frequency_sample_mean = frequency_sample_mean
        self.frequency_sample_std = frequency_sample_std

    def _create_weight_offset(self,
                              output_shape: tf.TensorShape,
                              intrinsic_weights: IntrinsicWeights,
                              name: str = None,
                              dtype: tf.dtypes.DType = tf.keras.backend.floatx()) -> tf.function:
        """Create RFF projection"""

        # RFF projection: multiply by these random frequencies before applying sinusoids
        # TODO: experiment with this initializer
        # rff_initializer = tf.initializers.RandomNormal(mean=self.frequency_sample_mean,
        #                                                stddev=self.frequency_sample_std)
        rff_initializer = tf.initializers.RandomUniform()
        rff_projection_matrix = tf.Variable(rff_initializer(shape=(intrinsic_weights.size, self.frequency_samples)),
                                            dtype=dtype,
                                            trainable=False,
                                            name=f'{name}_rff_projector')

        # create random projection matrix of RFF-upsampled features
        total_output_dim = 1
        for dim in output_shape:
            total_output_dim *= dim
        projection_matrix = tf.Variable(self.projection_matrix_initializer(shape=(2 * self.frequency_samples,
                                                                                  total_output_dim)),
                                        dtype=dtype,
                                        trainable=False,
                                        name=f'{name}_projector')

        # thunk to perform calculation; not a lambda b/c we want it to store an immediate calculation
        @tf.function
        def out_thunk():
            rff_projection = intrinsic_weights @ rff_projection_matrix
            return tf.reshape(tf.concat((tf.cos(rff_projection), tf.sin(rff_projection)), axis=1) @ projection_matrix,
                              shape=output_shape,
                              name=f'{name}_rff_offset')
        return out_thunk
