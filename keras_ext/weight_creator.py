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
from sklearn.random_projection import SparseRandomProjection
from scipy.sparse import find
import numpy as np


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


class SquaredTermsWeightCreator(WeightCreator):
    """Generates weights using a dense random linear projection and concats squared terms"""

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
        """Create dense weight projection with squared terms"""

        # create random projection matrix
        total_output_dim = 1
        for dim in output_shape:
            total_output_dim *= dim
        projection_matrix = tf.Variable(
            tf.concat((
                self.projection_matrix_initializer(shape=(intrinsic_weights.size, total_output_dim)),
                0.1 * self.projection_matrix_initializer(shape=(intrinsic_weights.size, total_output_dim)),
                0.01 * self.projection_matrix_initializer(shape=(intrinsic_weights.size, total_output_dim)),
            ), axis=0),
            dtype=dtype,
            trainable=False,
            name=f'{name}_projector')

        # return thunk that calculates the projection when called
        return tf.function(lambda: tf.reshape(tf.concat((intrinsic_weights.weights,
                                                         tf.math.square(intrinsic_weights.weights),
                                                         tf.math.pow(intrinsic_weights.weights, 3)), axis=1)
                                              @ projection_matrix,
                                              shape=output_shape,
                                              name=f'{name}_dense_offset'))


class RFFWeightCreator(WeightCreator):
    """Create random fourier feature projection -- first "frequency-samples" intrinsic dimension vector, then projects
    that onto full weight space"""

    def __init__(self,
                 initial_weight_initializer: tf.initializers.Initializer = 'glorot_normal',
                 projection_matrix_initializer: tf.initializers.Initializer = 'glorot_normal',
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
        rff_initializer = tf.initializers.RandomNormal(mean=self.frequency_sample_mean,
                                                       stddev=self.frequency_sample_std)
        # rff_initializer = tf.initializers.RandomUniform(minval=0, maxval=1)
        rff_projection_matrix = tf.Variable(rff_initializer(shape=(intrinsic_weights.size, self.frequency_samples)),
                                            dtype=dtype,
                                            trainable=False,
                                            name=f'{name}_rff_projector')

        # create random projection matrix of RFF-upsampled features
        total_output_dim = 1
        for dim in output_shape:
            total_output_dim *= dim
        projection_matrix = tf.Variable(self.projection_matrix_initializer(shape=(2 * self.frequency_samples + intrinsic_weights.size,
                                                                                  total_output_dim)),
                                        dtype=dtype,
                                        trainable=False,
                                        name=f'{name}_projector')

        # thunk to perform calculation
        @tf.function
        def out_thunk():
            rff_projection = intrinsic_weights @ rff_projection_matrix
            return tf.reshape(tf.concat((intrinsic_weights.weights,
                                         tf.cos(rff_projection),
                                         tf.sin(rff_projection)), axis=1) @ projection_matrix,
                              shape=output_shape,
                              name=f'{name}_rff_offset')

        return out_thunk


class SparseLinearWeightCreator(WeightCreator):
    """Sparse linear random projection"""
    """TODO: fix terrible performance"""

    def __init__(self,
                 initial_weight_initializer: tf.initializers.Initializer = 'glorot_normal',
                 projection_matrix_initializer: tf.initializers.Initializer = 'glorot_normal'):
        super().__init__(initial_weight_initializer)
        self.initial_weight_initializer = tf.initializers.get(initial_weight_initializer)
        self.projection_matrix_initializer = tf.initializers.get(projection_matrix_initializer)

    def _create_weight_offset(self,
                              output_shape: tf.TensorShape,
                              intrinsic_weights: IntrinsicWeights,
                              name: str = None,
                              dtype: tf.dtypes.DType = tf.keras.backend.floatx()) -> tf.function:
        """Create sparse weight projection -- copied from Uber repo with very little modification"""

        if dtype is None:
            dtype = tf.keras.backend.floatx()

        # Create projection matrix ww
        total_output_dim = 1
        for dim in output_shape:
            total_output_dim *= dim

        # Generate location and relative scale of non zero elements
        print('finding...')
        M = SparseRandomProjection(intrinsic_weights.size)._make_random_matrix(intrinsic_weights.size, total_output_dim)
        print('finding...')
        fm = find(M)
        print('finding...')

        # Create sparse projection matrix from small vv to full theta space
        ww0 = tf.SparseTensor(indices=np.array([fm[0], fm[1]]).T,
                              values=fm[2],
                              dense_shape=[intrinsic_weights.size, total_output_dim])

        # Create diagonal normalization matrix that will be filled in when all layers are created, so that we can normalize each
        # row of the projection matrix (with length equal to the total number of parameters in the model) once we have all its elements.
        # This will hold the norms of the rows of the un-normalized projection matrix.
        # normalizer = tf.Variable(tf.zeros(shape=intrinsic_weights.size, dtype=dtype),
        #                          trainable=False,
        #                          name='%s_normalizer' % name)

        # Pre-multiply the normalizer by the low-rank parameter vector to avoid a sparse matrix - sparse matrix product,
        # which is not well-supported in Tensorflow (instead of theta_full = (P*N^-1)*theta_small where P*N^-1 is a row-normalized
        # projection matrix, do P*(N^-1*theta_small)). (N^-1*theta_small) can be written as simply an element-wise vector division.
        # theta_small_norm = tf.divide(intrinsic_weights.weights, normalizer)

        # Compute delta from theta_0 using sparse projection
        # Note: sparse matrix must be first argument
        # delta_theta_flat = tf.sparse.sparse_dense_matmul(ww, theta_small_norm, adjoint_a=True, adjoint_b=True)

        # Create theta
        # theta_offset = tf.reshape(delta_theta_flat, output_shape)

        # Note: previous versions added only ww0 to _non_trainable_weights but skipped normalizer. Here we more correctly return both.
        # return theta_offset, [ww0]
        # return theta_offset, [ww0, normalizer]
        return tf.function(lambda: tf.reshape(
            tf.sparse.sparse_dense_matmul(tf.cast(ww0, dtype=dtype),
                                          tf.divide(intrinsic_weights.weights,
                                                    tf.math.sqrt(tf.sparse.reduce_sum(tf.cast(ww0, dtype=dtype), 1))),
                                          adjoint_a=True,
                                          adjoint_b=True),
            shape=output_shape))
