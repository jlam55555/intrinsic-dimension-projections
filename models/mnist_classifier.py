"""
Experimenting on MNIST
"""
import tensorflow as tf
from tensorflow.python.keras.regularizers import l2

from keras_ext.intrinsic_weights import IntrinsicWeights
from keras_ext.projection_layer import DenseRandomProjectionLayer
from keras_ext.weight_creator import WeightCreator


class MNISTClassifier:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.model: tf.keras.models.Model = None

    def build_fc_direct_model(self, layers=2, width=128, lr=0.001, initializer='he_uniform',
                              bias_regularizer=l2(1e-3), kernel_regularizer=l2(1e-3)):
        """Simple fully-connected model with ordinary keras layers"""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            *[tf.keras.layers.Dense(width,
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    bias_initializer=initializer,
                                    bias_regularizer=bias_regularizer,
                                    kernel_regularizer=kernel_regularizer)
              for _ in range(layers)],
            tf.keras.layers.Dense(10,
                                  kernel_initializer=initializer,
                                  bias_initializer=initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def build_projection_model(self,
                               intrinsic_weights: IntrinsicWeights,
                               weight_creator: WeightCreator,
                               layers=2,
                               width=128,
                               lr=0.001,
                               kernel_regularizer = l2(1e-3),
                               bias_regularizer = l2(1e-3)):
        """Fully-connected model with projection layers"""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            *[DenseRandomProjectionLayer(weight_creator,
                                         intrinsic_weights,
                                         width,
                                         activation='relu',
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer)
              for _ in range(layers)],
            DenseRandomProjectionLayer(weight_creator,
                                       intrinsic_weights,
                                       10,
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def build_cnn_direct_model(self):
        weight_decay = 0

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, init='he_normal', padding='valid', activation='relu',
                                   kernel_regularizer=l2(weight_decay)),
            tf.keras.layers.Conv2D(16, 3, 3, init='he_normal', padding='valid', activation='relu',
                                   kernel_regularizer=l2(weight_decay)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, 3, 3, init='he_normal', padding='valid', activation='relu',
                                   kernel_regularizer=l2(weight_decay)),
            tf.keras.layers.BatchNormalization(momentum=0.5),
            tf.keras.layers.Conv2D(16, 3, 3, init='he_normal', padding='valid', activation='relu',
                                   kernel_regularizer=l2(weight_decay)),  # (8, 8)
            tf.keras.layers.MaxPooling2D((2, 2)),  # (4, 4)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(800, kernel_initializer='he_normal', activation='relu',
                                  kernel_regularizer=l2(weight_decay)),
            tf.keras.layers.Dense(800, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)),
            tf.keras.layers.BatchNormalization(momentum=0.5),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(500, kernel_initializer='he_normal', activation='relu',
                                  kernel_regularizer=l2(weight_decay)),
            tf.keras.layers.Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def build_cnn_projection_model(self):
        # not implemented yet; haven't implemented batchnorm
        raise NotImplementedError

    def train(self, epochs=100):
        assert self.model is not None, 'Build model before training'
        return self.model.fit(self.x_train, self.y_train, epochs=epochs,
                              callbacks=[
                                  tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 0.999),
                                  tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', patience=10)
                              ])

    def evaluate(self):
        assert self.model is not None, 'Build model before evaluating'
        return self.model.evaluate(self.x_test, self.y_test)
