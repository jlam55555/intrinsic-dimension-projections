from models.mnist_classifier import MNISTClassifier

mnist_classifier = MNISTClassifier()
# mnist_classifier.build_fc_direct_model()
# mnist_classifier.model.summary()
# mnist_classifier.train(epochs=10)
# mnist_classifier.evaluate()

# things we can test:
# initializers, intrinsic dimension, learning rate, network parameters

# epochs = 10
# model_type = mnist_classifier.build_squared_fc_projection_model
# lr = 0.001
# initializer = 'glorot_uniform'
# intrinsic_dim = 200
#
# model_type(intrinsic_dim=200 + 100 * intrinsic_dim, lr=lr, initializer=initializer)
# print(
#     f'epochs: {epochs}; initializer: {initializer}; lr: {lr}; type: {"linear" if model_type == mnist_classifier.build_linear_fc_projection_model else "squared"}')
# mnist_classifier.model.summary()
# mnist_classifier.train(epochs=epochs)
# mnist_classifier.evaluate()
epochs = 1000
intrinsic_dims = range(6, 7)
initializers = ['glorot_uniform'] #['glorot_normal', 'glorot_uniform']
lrs = [0.001] #[0.1, 0.01, 0.001]
model_types = [mnist_classifier.build_linear_fc_projection_model,
               mnist_classifier.build_squared_fc_projection_model]

for intrinsic_dim in intrinsic_dims:
    for initializer in initializers:
        for lr in lrs:
            for model_type in model_types:
                model_type(width=200, intrinsic_dim=200+100*intrinsic_dim, lr=lr, initializer=initializer)
                print(f'epochs: {epochs}; initializer: {initializer}; lr: {lr}; type: {"linear" if model_type == mnist_classifier.build_linear_fc_projection_model else "squared"}')
                mnist_classifier.model.summary()
                mnist_classifier.train(epochs=epochs)
                mnist_classifier.evaluate()
