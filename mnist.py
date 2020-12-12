from keras_ext.intrinsic_weights import IntrinsicWeights
from keras_ext.weight_creator import DenseLinearWeightCreator, SquaredTermsWeightCreator, RFFWeightCreator
from models.mnist_classifier import MNISTClassifier
from datetime import datetime
import multiprocessing as mp
import numpy as np
import pickle

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

epochs = 100
intrinsic_dims = np.linspace(100, 1000, 10, dtype=int)
initializers = ['he_normal']
lrs = [0.001]
model_types = ['linear', 'power']


def run_model(model_type, epochs, initializer, lr):
    mnist_classifier = MNISTClassifier()

    intrinsic_weights = IntrinsicWeights(size=intrinsic_dim)

    if model_type == 'linear':
        weight_creator = DenseLinearWeightCreator(initial_weight_initializer=initializer,
                                                  projection_matrix_initializer='random_normal')
    elif model_type == 'power':
        weight_creator = SquaredTermsWeightCreator(initial_weight_initializer=initializer,
                                                   projection_matrix_initializer='random_normal',
                                                   squared_terms_coefficient=0.1,
                                                   cubed_terms_coefficient=0.01)
    elif model_type == 'rff':
        weight_creator = RFFWeightCreator(initial_weight_initializer=initializer,
                                          projection_matrix_initializer='random_normal',
                                          frequency_samples=intrinsic_dim // 2,
                                          frequency_sample_std=1,
                                          frequency_sample_mean=0)
    else:
        assert False, 'model_type must be one of [\'linear\', \'power\', \'rff\']'

    mnist_classifier.build_projection_model(intrinsic_weights,
                                            weight_creator,
                                            width=200,
                                            lr=lr)
    print(f'epochs: {epochs}; intrinsic_dim: {intrinsic_dim}; initializer: {initializer}; lr: {lr}; type: {model_type}')
    summary_str = ''
    def print_fn(line):
        nonlocal summary_str
        summary_str += line + '\n'
    mnist_classifier.model.summary(print_fn=print_fn)
    hist = mnist_classifier.train(epochs=epochs)
    eval = mnist_classifier.evaluate()

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
        'timestamp': timestamp
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
                process_eval = mp.Process(target=run_model, args=(model_type, epochs, initializer, lr))
                process_eval.start()
                process_eval.join()
