from models.spiral_classifier import SpiralClassifier

import tensorflow as tf

tf.config.run_functions_eagerly(False)

spiral_classifier = SpiralClassifier(N=500, sigma=0.25)
# spiral_classifier.plot()

# spiral_classifier.build_direct_model()
# spiral_classifier.train(epochs=100)
# spiral_classifier.evaluate()
# spiral_classifier.model.summary()

spiral_classifier.build_projection_model(intrinsic_dim=100)
spiral_classifier.train(epochs=5000)
spiral_classifier.evaluate()
spiral_classifier.model.summary()
