from models.mnist_classifier import MNISTClassifier
import multiprocessing as mp
import numpy as np
import pickle

epochs = 100
intrinsic_dims = np.linspace(100, 1000, 10, dtype=int)
initializers = ['glorot_uniform']
lrs = [0.001]
model_types = ['linear', 'squared']


def run_model(model_type, epochs, initializer, lr):
    mnist_classifier = MNISTClassifier()

    if model_type == 'linear':
        creator = mnist_classifier.build_linear_fc_projection_model
    else:
        creator = mnist_classifier.build_squared_fc_projection_model

    creator(width=200, intrinsic_dim=intrinsic_dim, lr=lr, initializer=initializer)
    print(f'epochs: {epochs}; initializer: {initializer}; lr: {lr}; type: {model_type}')
    summary_str = ''
    def print_fn(line):
        nonlocal summary_str
        summary_str += line + '\n'
    mnist_classifier.model.summary(print_fn=print_fn)
    hist = mnist_classifier.train(epochs=epochs)
    eval = mnist_classifier.evaluate()

    # write results to file; write this for every model training to safeguard against OOM error
    # using multiple processes *should* fix this but not sure
    out_obj = {
        'model_type': model_type,
        'epochs': epochs,
        'initializer': initializer,
        'lr': lr,
        'intrinsic_dim': intrinsic_dim,
        'history': hist.history,
        'eval': eval,
        'summary': summary_str
    }
    out_filename = f'runs/mnist_{model_type}_{intrinsic_dim}.pkl'
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
