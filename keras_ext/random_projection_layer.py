"""
Random projection layers form the core of this project. The weights theta_D (w/ cardinality D) for each layer are the
sum of a non-trainable set of initial weights theta_0, as well as an offset theta_off. theta_off is generated from an
OffsetCreator instance, which projects a smaller trainable weight matrix theta_d (w/ cardinality d < D) onto theta_off.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import _minimize
from tensorflow.python.layers.base import InputSpec

from keras_ext.intrinsic_weights import IntrinsicWeights
from keras_ext.offset_creator import OffsetCreator


class RandomProjectionLayer(Layer):
    """Version of Layer that randomly projects a lower-dimensional space onto the full-dimensional weights"""

    def __init__(self,
                 offset_creator: OffsetCreator,
                 intrinsic_weights: IntrinsicWeights):
        """
        Each RandomProjectionLayer instance is associated with an offset creator, and thus a lower-dimensional
        trainable space
        """
        super(RandomProjectionLayer, self).__init__()
        self.offset_creator = offset_creator
        self.intrinsic_weights = intrinsic_weights

    def add_weight(self,
                   name=None,
                   shape=None,
                   dtype=tf.keras.backend.floatx(),
                   initializer=None,
                   regularizer=None,
                   trainable=None,
                   constraint=None,
                   partitioner=None,
                   use_resource=None,
                   synchronization=tf.VariableSynchronization.AUTO,
                   aggregation=tf.VariableAggregation.NONE,
                   **kwargs):
        """
        Version of add_weight that creates both the non-trainable initial weights theta_0 and the offset theta_off
        """
        initializer = tf.keras.initializers.get(initializer)
        regularizer = tf.keras.regularizers.get(regularizer)

        weight_graph = self.offset_creator.create_weight(self.intrinsic_weights,
                                                         name=f'{name}_weight',
                                                         output_shape=shape)

        return weight_graph

        # create non-trainable initial weights, theta_0
        # theta_0 = tf.Variable(initializer(shape=shape),
        #                       trainable=False,
        #                       dtype=dtype,
        #                       name=f'{name}_theta0')
        #
        # # create trainable offsets, theta_t
        # theta_off = self.offset_creator.create_offset(intrinsic_weights=self.intrinsic_weights,
        #                                               name=f'{name}_offset_weight',
        #                                               output_shape=shape)

        # total weights are the sum of the initial weights and the offsets
        # theta = tf.add(theta_0, theta_off)
        # # theta = theta_0 + theta_0
        #
        # if regularizer is not None:
        #     self.add_loss(regularizer(theta))
        #
        # print(theta_0, theta)
        #
        # return theta

        # return theta_0, theta_off


class DenseRandomProjectionLayer(RandomProjectionLayer):
    """Dense version of random projection layer"""

    def __init__(self,
                 offset_creator: OffsetCreator,
                 intrinsic_weights: IntrinsicWeights,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(DenseRandomProjectionLayer, self).__init__(offset_creator, intrinsic_weights)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_graph = None
        self.kernel_graph = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel_graph = self.add_weight(shape=(input_dim, self.units),
                                            name='kernel',
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias_graph = self.add_weight(shape=(self.units,),
                                              name='bias',
                                              initializer=self.bias_initializer,
                                              regularizer=self.bias_regularizer)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        output = inputs @ self.kernel_graph
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias_graph)
        if self.activation is not None:
            output = self.activation(output)
        return output
