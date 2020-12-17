"""
Toy dataset to classify which spiral a point lies in.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from keras_ext.intrinsic_weights import IntrinsicWeights
from keras_ext.projection_layer import DenseRandomProjectionLayer
from keras_ext.weight_creator import DenseLinearWeightCreator, RFFWeightCreator, SquaredTermsWeightCreator


class SpiralClassifier:
    """Generate spirals"""

    def __init__(self, N=250, sigma=0.5):
        """Generate dataset"""

        # generate spirals
        thetas = np.random.uniform(low=np.pi, high=3.5 * np.pi, size=(N, 2))
        noise = np.random.normal(loc=0, scale=sigma, size=(N, 2))

        spiral_1_x = (thetas[:, 0] + noise[:, 0]) * np.cos(thetas[:, 0])
        spiral_1_y = (thetas[:, 0] + noise[:, 0]) * np.sin(thetas[:, 0])
        self.spiral_1 = np.vstack((spiral_1_x, spiral_1_y)).T

        spiral_2_x = (thetas[:, 1] + noise[:, 1]) * np.cos(np.pi + thetas[:, 1])
        spiral_2_y = (thetas[:, 1] + noise[:, 1]) * np.sin(np.pi + thetas[:, 1])
        self.spiral_2 = np.vstack((spiral_2_x, spiral_2_y)).T

        # split into train/test
        spirals = np.vstack((self.spiral_1, self.spiral_2))
        labels = np.vstack((np.zeros(shape=(N, 1)), np.ones(shape=(N, 1))))

        dataset = np.hstack((spirals, labels))
        np.random.shuffle(dataset)

        split = int(0.8 * N)
        self.x_train = dataset[:split, 0:2]
        self.x_test = dataset[split:, 0:2]
        self.y_train = dataset[:split, 2][:, np.newaxis]
        self.y_test = dataset[split:, 2][:, np.newaxis]

        self.model: tf.keras.Model = None

    def plot(self):
        """Plot spirals"""
        plt.scatter(self.spiral_1[:, 0], self.spiral_1[:, 1])
        plt.scatter(self.spiral_2[:, 0], self.spiral_2[:, 1])
        plt.show()

    def build_direct_model(self):
        model = tf.keras.models.Sequential(layers=[tf.keras.layers.Input(shape=(2,)),
                                                   tf.keras.layers.Dense(5),
                                                   tf.keras.layers.ReLU(),
                                                   tf.keras.layers.Dense(5),
                                                   tf.keras.layers.ReLU(),
                                                   tf.keras.layers.Dense(2)])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.model = model

    def build_projection_model(self, intrinsic_dim=100):
        intrinsic_weights = IntrinsicWeights(intrinsic_dim,
                                             initializer='glorot_normal')
        weight_creator = DenseLinearWeightCreator(initial_weight_initializer='glorot_uniform',
                                                  projection_matrix_initializer='glorot_uniform')

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            DenseRandomProjectionLayer(weight_creator, intrinsic_weights, 32,
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.0001),
                                       bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.0001)),
            tf.keras.layers.ReLU(),
            DenseRandomProjectionLayer(weight_creator, intrinsic_weights, 32,
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.0001),
                                       bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.0001)),
            tf.keras.layers.ReLU(),
            DenseRandomProjectionLayer(weight_creator, intrinsic_weights, 2,
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.0001),
                                       bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.0001))
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.model = model

    def train(self, epochs=100):
        assert self.model is not None

        # tensorboard logging
        # see: https://www.tensorflow.org/tensorboard/graphs
        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model.fit(self.x_train, self.y_train, epochs=epochs,
                       callbacks=[
                           # tf.keras.callbacks.TensorBoard(log_dir=logdir),
                                  tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 0.999)])

    def evaluate(self):
        assert self.model is not None
        self.model.evaluate(self.x_test, self.y_test)
