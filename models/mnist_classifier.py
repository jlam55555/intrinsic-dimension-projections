"""
Experimenting on MNIST
"""
import tensorflow as tf

from keras_ext.intrinsic_weights import IntrinsicWeights
from keras_ext.projection_layer import DenseRandomProjectionLayer
from keras_ext.weight_creator import DenseLinearWeightCreator


class MNISTClassifier:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.model: tf.keras.models.Model = None

    def build_fc_direct_model(self, layers=2, width=128):
        """Simple fully-connected model with ordinary keras layers"""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            *[tf.keras.layers.Dense(width, activation='relu') for _ in range(layers)],
            tf.keras.layers.Dense(10)
        ])

    def build_fc_projection_model(self, layers=2, width=128, intrinsic_dim=200):
        """Fully-connected model with projection layers"""
        intrinsic_weights = IntrinsicWeights(size=intrinsic_dim)
        weight_creator = DenseLinearWeightCreator(initial_weight_initializer='glorot_normal',
                                                  projection_matrix_initializer='glorot_normal')

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            *[DenseRandomProjectionLayer(weight_creator, intrinsic_weights, width, activation='relu')
              for _ in range(layers)],
            tf.keras.layers.Dense(10)
        ])

    def build_cnn_direct_model(self):
        tf.keras.models.Sequential()
        xx = Convolution2D(16, kernel_size=3, strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = Convolution2D(16, 3, 3, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Convolution2D(16, 3, 3, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = BatchNormalization(momentum=0.5)(xx)
        xx = Convolution2D(16, 3, 3, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)  # (8, 8)
        xx = MaxPooling2D((2, 2))(xx)  # (4, 4)
        xx = Flatten()(xx)
        xx = Dense(800, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dense(800, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = BatchNormalization(momentum=0.5)(xx)
        xx = Activation('relu')(xx)
        xx = Dense(500, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        pass

    def build_cnn_projection_model(self):
        pass

    def train(self, epochs=100):
        assert self.model is not None
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.model.fit(self.x_train, self.y_train, epochs=epochs)

    def evaluate(self):
        assert self.model is not None
        self.model.evaluate(self.x_test, self.y_test)
