import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

# direct vs. Li et al. vs. ours
# mnist_files = glob.glob('*')
#
# # from: https://github.com/uber-research/intrinsic-dimension/blob/master/intrinsic_dim/plots/main_plots.ipynb (cell 11)
# mnist_subspace_mlp_L2 = np.array([ [10, 0.1581],[50, 0.4555],[100, 0.6055],[200, 0.7713],[300, 0.8181],[350, 0.8328],[375, 0.8589],[400, 0.8538],[425, 0.8586],[450, 0.8593],[475, 0.8687],[500, 0.8691],[525, 0.8816],[550, 0.8851],[575, 0.8808],[600, 0.888],[625, 0.8899],[650, 0.8931],[675, 0.8945],[700, 0.8982],[725, 0.8967],[750, 0.9022],[775, 0.9003],[800, 0.908],[850, 0.9073],[900, 0.9132],[1000, 0.9182],[1250, 0.9292],[1500, 0.9322],[1750, 0.9219],[2000, 0.9199],[2250, 0.9263],[2500, 0.9295],[3000, 0.931],[4000, 0.9351],[5000, 0.9443],[5500, 0.9471],[6000, 0.9503],[6500, 0.9505],[7000, 0.9508],[7500, 0.9518],[8000, 0.954],[8500, 0.9556],[9000, 0.9574],[9500, 0.9592],[10000, 0.9567],[12500, 0.9657],[15000, 0.9685],[17500, 0.97],[20000, 0.97],[22500, 0.9722],[25000, 0.972] ])
#
# mnist_subspace_mlp_L2 = mnist_subspace_mlp_L2[mnist_subspace_mlp_L2[:, 0] <= 1000]
#
# plt.plot(mnist_subspace_mlp_L2[:, 0], mnist_subspace_mlp_L2[:, 1], 'o-')
# plt.show()

# regular vs. power
mnist_files = glob.glob('runs/mnist_nonnormalized_p_/mnist*.pkl')
mnist_files2 = glob.glob('runs/many_runs/mnist*.pkl')
results1 = []   # linear dense projection
results2 = []   # power dense projection
results3 = []   # rff
for filename in (mnist_files + mnist_files2):
    with open(filename, 'rb') as file_handle:
        model_dict = pickle.load(file_handle)

    if model_dict['model_type'] == 'linear':
        results1.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
    elif model_dict['model_type'] == 'power':
        results2.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
    else:
        results3.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])

results = np.array(results1)
plt.scatter(results[:, 0], results[:, 1])
results = np.array(results2)
plt.scatter(results[:, 0], results[:, 1])
results = np.array(results3)
plt.scatter(results[:, 0], results[:, 1])
plt.ylabel('Accuracy')
plt.xlabel('Intrinsic dimension')

# power w/ different coefficients
# mnist_files = glob.glob('runs/power_coefficients/mnist*.pkl')
# results = []
# for filename in mnist_files:
#     with open(filename, 'rb') as file_handle:
#         model_dict = pickle.load(file_handle)
#
#     results.append([model_dict['squared_coefficient'] if 'squared_coefficient' in model_dict else 0.1,
#                     model_dict['cubed_coefficient'] if 'cubed_coefficient' in model_dict else 0.01,
#                     model_dict['intrinsic_dim'],
#                     model_dict['eval'][1]])
#
# results = np.array(results)
# experiments = [
#     (1, 1),
#     (0.5, 0.25),
#     (0.1, 0.01),
#     (0.01, 0.001),
#     (1, 0),
#     (0.5, 0),
#     (0.1, 0)
# ]
# plt.figure()
# for lambda_s, lambda_c in experiments:
#     rows = (results[:, 0] == lambda_s) & (results[:, 1] == lambda_c)
#     plt.scatter(results[rows, 2], results[rows, 3])
# plt.title('784-200-200 FC MNIST Accuracy vs. Intrinsic Dimension for Power Types')
# plt.ylabel('Accuracy')
# plt.xlabel('Intrinsic dimension')
# plt.legend([f'$\lambda_s: {lambda_s}; \lambda_c: {lambda_c}$' for lambda_s, lambda_c in experiments])
#
plt.show()
