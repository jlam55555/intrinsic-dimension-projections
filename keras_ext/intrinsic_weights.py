"""
Intrinsic weights are the set of d weights that get projected up onto the D weights of the network. One set of intrinsic
weights is needed for each network and the same instance is used for all its layers.

TODO: perhaps the above statement shouldn't be true? Maybe different intrinsic weights for different layers/layer types?
"""

import tensorflow as tf


class IntrinsicWeights:
    """Set of (trainable) intrinsic weights that get projected onto the weight offsets"""

    def __init__(self, size, initializer=tf.initializers.RandomNormal()):
        """Creates a set of intrinsic weights; shape is irrelevant, for simplicity use column vector"""
        initializer = tf.initializers.get(initializer)
        self.weights = tf.Variable(initializer(shape=(1, size)),
                                   name='intrinsic_weights')
        self.size = size

    def __add__(self, other: tf.Variable) -> tf.Variable:
        return self.weights + other

    def __matmul__(self, other: tf.Variable) -> tf.Variable:
        return self.weights @ other
