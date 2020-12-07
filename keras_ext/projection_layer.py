"""
Versions of the tf.keras.layers Layers for use with intrinsic weight projection.
"""
from typing import Callable

import tensorflow as tf
from tensorflow.python.layers.base import InputSpec

from keras_ext.intrinsic_weights import IntrinsicWeights
from keras_ext.weight_creator import WeightCreator


class ProjectionLayer(tf.keras.layers.Layer):
    """Version of Layer that projects a lower-dimensional space onto the full layer weights using WeightCreator"""

    def __init__(self,
                 weight_creator: WeightCreator,
                 intrinsic_weights: IntrinsicWeights,
                 **kwargs):
        super().__init__(**kwargs)
        self.weight_creator = weight_creator
        self.intrinsic_weights = intrinsic_weights

        # the all trainable tf.Variable objects have to be class attributes of the model in order
        # for them to be trained -- took a weekend to find this out
        self.intrinsic_weights_variable = intrinsic_weights.weights

    def add_weight(self,
                   name=None,
                   shape=None,
                   dtype=None,
                   initializer=tf.initializers.RandomNormal(),
                   regularizer=None,
                   trainable=None,
                   constraint=None,
                   partitioner=None,
                   use_resource=None,
                   synchronization=tf.VariableSynchronization.AUTO,
                   aggregation=tf.VariableAggregation.NONE,
                   **kwargs) -> Callable[[], tf.Variable]:
        """Version of add_weight that creates the weight using weight_creator"""
        return self.weight_creator.create_weight(shape, self.intrinsic_weights, name, dtype)


class DenseRandomProjectionLayer(ProjectionLayer):
    """Projection version of dense layer"""

    def __init__(self,
                 weight_creator: WeightCreator,
                 intrinsic_weights: IntrinsicWeights,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super().__init__(weight_creator, intrinsic_weights, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_thunk = self.kernel_thunk = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel_thunk = self.add_weight(shape=(input_dim, self.units),
                                            name='kernel',
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias_thunk = self.add_weight(shape=(self.units,),
                                              name='bias',
                                              initializer=self.bias_initializer,
                                              regularizer=self.bias_regularizer)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        kernel = self.kernel_thunk()
        bias = self.bias_thunk()
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(kernel))
        if self.bias_regularizer is not None:
            self.add_loss(self.bias_regularizer(bias))

        output = inputs @ kernel
        if self.use_bias:
            output = tf.nn.bias_add(output, bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
