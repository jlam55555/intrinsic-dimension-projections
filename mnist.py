from models.mnist_classifier import MNISTClassifier

mnist_classifier = MNISTClassifier()
# mnist_classifier.build_fc_direct_model()
# mnist_classifier.model.summary()
# mnist_classifier.train(epochs=10)
# mnist_classifier.evaluate()

# things we can test:
# initializers, intrinsic dimension, learning rate, network parameters

epochs = 1000
for intrinsic_dim in range(5):
    for initializer in ['glorot_normal', 'glorot_uniform']:
        for lr in [0.1, 0.01, 0.001]:
            for model_type in [mnist_classifier.build_linear_fc_projection_model,
                               mnist_classifier.build_squared_fc_projection_model]:
                model_type(intrinsic_dim=200+100*intrinsic_dim, lr=lr, initializer=initializer)
                print(f'epochs: {epochs}; initializer: {initializer}; lr: {lr}; type: {"linear" if model_type == mnist_classifier.build_linear_fc_projection_model else "squared"}')
                mnist_classifier.model.summary()
                mnist_classifier.train(epochs=epochs)
                mnist_classifier.evaluate()
