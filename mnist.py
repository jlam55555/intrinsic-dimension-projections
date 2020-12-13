from keras_ext.intrinsic_weights import IntrinsicWeights
from keras_ext.projection_layer import DenseRandomProjectionLayer
from keras_ext.weight_creator import DenseLinearWeightCreator, SquaredTermsWeightCreator, RFFWeightCreator
from models.mnist_classifier import MNISTClassifier
from datetime import datetime
import multiprocessing as mp
import numpy as np
import pickle

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

epochs = 100
intrinsic_dims = np.linspace(100, 1000, 10, dtype=int)
initializers = ['he_normal']
lrs = [0.001]
model_types = ['power']
train_proj = True


def run_model(model_type, epochs, initializer, lr, train_proj: bool = False):
    mnist_classifier = MNISTClassifier()

    intrinsic_weights = IntrinsicWeights(size=intrinsic_dim)

    if model_type == 'linear':
        weight_creator = DenseLinearWeightCreator(initial_weight_initializer=initializer,
                                                  projection_matrix_initializer='random_normal',
                                                  trainable_proj=train_proj)
    elif model_type == 'power':
        weight_creator = SquaredTermsWeightCreator(initial_weight_initializer=initializer,
                                                   projection_matrix_initializer='random_normal',
                                                   squared_terms_coefficient=1,
                                                   cubed_terms_coefficient=1,
                                                   trainable_proj=train_proj)
    elif model_type == 'rff':
        weight_creator = RFFWeightCreator(initial_weight_initializer=initializer,
                                          projection_matrix_initializer='random_normal',
                                          frequency_samples=intrinsic_dim // 2,
                                          frequency_sample_std=1,
                                          frequency_sample_mean=0,
                                          trainable_proj=train_proj)
    else:
        assert False, 'model_type must be one of [\'linear\', \'power\', \'rff\']'

    mnist_classifier.build_projection_model(intrinsic_weights,
                                            weight_creator,
                                            width=200,
                                            lr=lr)

    # save projection matrices before
    projection_matrices_before = []
    if train_proj:
        for layer in mnist_classifier.model.layers:
            if isinstance(layer, DenseRandomProjectionLayer):
                projection_matrices_before.append([
                    layer.trainable_weight1.numpy() if layer.trainable_weight1 is not None else None,
                    layer.trainable_weight2.numpy() if layer.trainable_weight2 is not None else None,
                    layer.trainable_weight3.numpy() if layer.trainable_weight3 is not None else None,
                    layer.trainable_weight4.numpy() if layer.trainable_weight4 is not None else None,
                ])

    # get untrained weights
    untrained_intrinsic_weights = intrinsic_weights.weights.numpy()

    print(f'epochs: {epochs}; intrinsic_dim: {intrinsic_dim}; initializer: {initializer}; lr: {lr}; type: {model_type}')
    summary_str = ''
    def print_fn(line):
        nonlocal summary_str
        summary_str += line + '\n'
    mnist_classifier.model.summary(print_fn=print_fn)
    hist = mnist_classifier.train(epochs=epochs)
    eval = mnist_classifier.evaluate()

    # save projection matrices after
    projection_matrices_after = []
    if train_proj:
        for layer in mnist_classifier.model.layers:
            if isinstance(layer, DenseRandomProjectionLayer):
                projection_matrices_after.append([
                    layer.trainable_weight1.numpy() if layer.trainable_weight1 is not None else None,
                    layer.trainable_weight2.numpy() if layer.trainable_weight2 is not None else None,
                    layer.trainable_weight3.numpy() if layer.trainable_weight3 is not None else None,
                    layer.trainable_weight4.numpy() if layer.trainable_weight4 is not None else None,
                ])

    # TODO: remove this line; for debugging
    print(projection_matrices_before, projection_matrices_after)

    # write results to file; write this for every model training to safeguard against OOM error
    # using multiple processes *should* fix this but not sure
    timestamp = datetime.now().strftime("%y-%m-%d-%H%M")
    out_obj = {
        'model_type': model_type,
        'epochs': epochs,
        'initializer': initializer,
        'lr': lr,
        'squared_coefficient': weight_creator.squared_terms_coefficient if model_type == 'power' else None,
        'cubed_coefficient': weight_creator.cubed_terms_coefficient if model_type == 'power' else None,
        'intrinsic_dim': intrinsic_dim,
        'history': hist.history,
        'eval': eval,
        'summary': summary_str,
        'timestamp': timestamp,
        'train_proj': train_proj,
        'projection_before': projection_matrices_before,
        'projection_after': projection_matrices_after,
        'intrinsic_weights_before': untrained_intrinsic_weights,
        'intrinsic_weights_after': intrinsic_weights.weights.numpy()
    }
    out_filename = f'runs/mnist_normalized_{model_type}_{intrinsic_dim}_{timestamp}.pkl'
    with open(out_filename, 'wb') as out_handle:
        pickle.dump(out_obj, out_handle)


for intrinsic_dim in intrinsic_dims:
    for initializer in initializers:
        for lr in lrs:
            for model_type in model_types:
                # see: https://github.com/tensorflow/tensorflow/issues/36465#issuecomment-582749350
                # run each model in a new process so that memory gets cleaned up
                process_eval = mp.Process(target=run_model, args=(model_type, epochs, initializer, lr, train_proj))
                process_eval.start()
                process_eval.join()
