from models.mnist_classifier import MNISTClassifier

mnist_classifier = MNISTClassifier()
mnist_classifier.build_fc_direct_model()
mnist_classifier.train(epochs=10)
mnist_classifier.evaluate()

mnist_classifier.build_fc_projection_model()
mnist_classifier.train(epochs=10)
mnist_classifier.evaluate()
