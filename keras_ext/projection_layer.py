"""
Versions of the tf.keras.layers Layers for use with intrinsic weight projection.
"""
from typing import Callable

import tensorflow as tf
from tensorflow.python.keras.backend import conv3d, conv2d, conv1d, bias_add
from tensorflow.python.keras.utils import conv_utils
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
        self.bias_thunk = self.kernel_thunk = lambda: None

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


class _ConvProjectionLayer(ProjectionLayer):
    """Abstract nD convolution layer (private, used as implementation base). Only the intrinsic parameters are
    trainable. Largely copied from the Uber implementation"""

    def __init__(self,
                 weight_creator: WeightCreator,
                 intrinsic_weights: IntrinsicWeights,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(weight_creator, intrinsic_weights, **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.kernel_thunk = self.bias_thunk = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel_thunk = self.add_weight(shape=kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_thunk = self.add_weight(shape=(self.filters,),
                                              initializer=self.bias_initializer,
                                              name='bias',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)
        else:
            self.bias_thunk = lambda: None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        if self.rank == 1:
            outputs = conv1d(
                inputs,
                self.kernel_thunk(),
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = conv2d(
                inputs,
                self.kernel_thunk(),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = conv3d(
                inputs,
                self.kernel_thunk(),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = bias_add(
                outputs,
                self.bias_thunk(),
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)


class Conv2DProjectionLayer(_ConvProjectionLayer):
    '''Low Rank Basis Conv2D. Filters if number of filters, output dimension is filters

    Largely copied from Uber implementation'''

    def __init__(self,
                 weight_creator: WeightCreator,
                 intrinsic_weights: IntrinsicWeights,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(weight_creator=weight_creator,
                         intrinsic_weights=intrinsic_weights,
                         rank=2,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         data_format=data_format,
                         dilation_rate=dilation_rate,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.input_spec = InputSpec(ndim=4)
