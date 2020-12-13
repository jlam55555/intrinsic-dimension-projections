import matplotlib.pyplot as plt
import pickle
import glob
import numpy as np

files = glob.glob('runs/trainable_proj/mnist_normalized_power_400*.pkl')
for filename in files:
    with open(filename, 'rb') as file_handle:
        model_dict = pickle.load(file_handle)

    before = model_dict['projection_before']
    after = model_dict['projection_after']

    intrinsic_weights_before = model_dict['intrinsic_weights_before']
    intrinsic_weights_after = model_dict['intrinsic_weights_after']

    before_flat = []
    for layer in before:
        for weight in layer:
            if weight is not None:
                before_flat.append(weight.flatten())
    before_flat = np.concatenate(before_flat)

    after_flat = []
    after_flat1 = []
    after_flat2 = []
    after_flat3 = []
    for layer in after:
        for weight in layer:
            if weight is not None:
                after_flat.append(weight.flatten())
                l = int(weight.shape[1]/3)
                after_flat1.append(weight[:,0:l].flatten())
                after_flat2.append(weight[:,l:2*l].flatten())
                after_flat3.append(weight[:,2*l:].flatten())

    after_flat = np.concatenate(after_flat).flatten()
    after_flat1 = np.concatenate(after_flat1).flatten()
    after_flat2 = np.concatenate(after_flat2).flatten()
    after_flat3 = np.concatenate(after_flat3).flatten()

    print(before_flat.std(), after_flat.std(), after_flat1.std(), after_flat2.std(), after_flat3.std())

    print(intrinsic_weights_before.std(), intrinsic_weights_after.std())

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.hist(after_flat, bins=np.linspace(-0.5, 0.5, 100))
    plt.title('After')

    plt.subplot(2, 3, 2)
    plt.hist(after_flat1, bins=np.linspace(-0.5, 0.5, 100))
    plt.title('1')

    plt.subplot(2, 3, 3)
    plt.hist(after_flat2, bins=np.linspace(-0.5, 0.5, 100))
    plt.title('2')

    plt.subplot(2, 3, 4)
    plt.hist(after_flat3, bins=np.linspace(-0.5, 0.5, 100))
    plt.title('3')

    plt.subplot(2, 3, 5)
    plt.hist(intrinsic_weights_before.flatten(), bins=np.linspace(-0.25, 0.25, 100))
    plt.title('4')

    plt.subplot(2, 3, 6)
    plt.hist(intrinsic_weights_after.flatten(), bins=np.linspace(-0.025, 0.025, 100))
    plt.title('5')

    plt.show()

    # plt.hist(np.array([np.array([weight.flatten() for weight in layer]).flatten() for layer in before]))
    # plt.plot()
