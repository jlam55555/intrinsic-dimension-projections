"""
Random projection layers form the core of this project. The weights theta_D (w/ cardinality D) for each layer are the
sum of a non-trainable set of initial weights theta_0, as well as an offset theta_off. theta_off is generated from an
OffsetCreator instance, which projects a smaller trainable weight matrix theta_d (w/ cardinality d < D) onto theta_off.
"""
from typing import Type

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer

from keras_ext.offset_creator import OffsetCreator


class RandomProjectionLayer(Layer):
    """Version of Layer that randomly projects a lower-dimensional space onto the full-dimensional weights"""

    def __init__(self,
                 offset_creator_class: Type[OffsetCreator] = None):
        """
        Each RandomProjectionLayer instance is associated with an offset creator, and thus a lower-dimensional
        trainable space
        """
        super(RandomProjectionLayer, self).__init__()
        self.offset_creator = offset_creator_class()

    def add_weight(self,
                   name=None,
                   shape=None,
                   dtype=None,
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
        Version of add_weight that creates both the non-trainable initial weights theta_0 and the offset theta_t
        """
        initializer = initializers.get(initializer)
        if dtype is None:
            # get default float type
            dtype = tf.keras.backend.floatx()

        # create non-trainable initial weights, theta_0
        theta_0 = tf.Variable(initializer(shape),
                              trainable=False,
                              dtype=dtype,
                              name=f'{name}_theta0')

        # create trainable offsets, theta_t
        theta_off = self.offset_creator.test()
