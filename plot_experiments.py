import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
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

#set fonts
font = {'family': 'cmr10', 'size': 18}
mpl.rc('font', **font)  # change the default font to Computer Modern Roman
mpl.rcParams['axes.unicode_minus'] = False  # because cmr10 does not have a Unicode minus sign

# # assumes X, Y are Nx1, Y is Nx1
# def LinReg(X, Y):
#     if X.ndim == 1:
#         X = X[:, np.newaxis]
#     if Y.ndim == 1:
#         Y = Y[:, np.newaxis]
#     print(X.shape, Y.shape)
#     # augment X with set of 1's
#     X = np.concatenate((np.ones_like(X), X), axis=1)
#     return np.linalg.pinv(X.T @ X) @ X.T @ Y
#
# # regular vs. power
# plt.figure(figsize=(8,6))
# mnist_files = glob.glob('runs/many_runs/mnist*.pkl')
# mnist_files2 = glob.glob('runs/_initialized2/mnist*.pkl')
# results1 = []   # linear dense projection
# results2 = []   # power dense projection
# results3 = []   # rff
# for filename in (mnist_files + mnist_files2):
#     with open(filename, 'rb') as file_handle:
#         model_dict = pickle.load(file_handle)
#
#     if model_dict['model_type'] == 'linear':
#         results1.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
#     elif model_dict['model_type'] == 'power':
#         results2.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
#     else:
#         results3.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
#
# results = np.array(results1)
# # plt.scatter(results[:, 0], results[:, 1], c='b', alpha=0.2, marker='x')
# # b, m = LinReg(1/results[:, 0], results[:, 1])
# # x = np.arange(1, 1000)
# # y = m/x + b
# # plt.plot(x, y, c='b')
# # get means for every intrinsic dimension value
# ids = np.linspace(100, 1000, 10)
# means = np.zeros((10, ))
# stds = np.zeros((10, ))
# for i, id in enumerate(ids):
#     means[i] = np.mean(results[results[:, 0] == id, 1])
#     stds[i] = np.std(results[results[:, 0] == id, 1])
# plt.plot(ids, means, 'b.-')
# plt.errorbar(ids, means, yerr=stds, c='b', capsize=5, elinewidth=1)
#
# # overall_accuracy = 0.9799
# data_mnist_mlp_2_200_standard = np.array([
#     [10, 0.2023],[50, 0.4345],[100, 0.6124],[200, 0.7551],[300, 0.8189],[350, 0.8347],[375, 0.8565],[400, 0.8516],[425, 0.8596],[450, 0.8578],[475, 0.8621],[500, 0.8693],[525, 0.8803],[550, 0.8787],[575, 0.8789],[600, 0.886],[625, 0.8935],[650, 0.8889],[675, 0.892],[700, 0.8932],[725, 0.901],[750, 0.9001],[775, 0.8966],[800, 0.9004],[850, 0.9081],[900, 0.9104],[1000, 0.9163],[1250, 0.9276],[1500, 0.9302]
# ])
# plt.plot(data_mnist_mlp_2_200_standard[:, 0], data_mnist_mlp_2_200_standard[:, 1], 'g.-')
# plt.legend(['Local 784-200-200', 'Li et. al (2018) 784-200-200'])
#
# # results = np.array(results2)
# # # b, m = LinReg(1/results[:, 0], results[:, 1])
# # # x = np.arange(1, 1000)
# # # y = m/x + b
# # # plt.plot(x, y, c='r')
# # # plt.scatter(results[:, 0], results[:, 1], c='r', alpha=0.2, marker='x')
# # # get means for every intrinsic dimension value
# # ids = np.linspace(100, 1000, 10)
# # means = np.zeros((10, ))
# # stds = np.zeros((10, ))
# # for i, id in enumerate(ids):
# #     means[i] = np.mean(results[results[:, 0] == id, 1])
# #     stds[i] = np.std(results[results[:, 0] == id, 1])
# # plt.plot(ids, means, 'r.-')
# # # plt.errorbar(ids, means, yerr=stds, c='r', capsize=5, elinewidth=1)
#
# # results = np.array(results3)
# # # b, m = LinReg(1/results[:, 0], results[:, 1])
# # # x = np.arange(1, 1000)
# # # y = m/x + b
# # # plt.plot(x, y, c='k')
# # # get means for every intrinsic dimension value
# # # plt.scatter(results[:, 0], results[:, 1], c='k', alpha=0.2, marker='x')
# # ids = np.linspace(100, 1000, 10)
# # means = np.zeros((10, ))
# # for i, id in enumerate(ids):
# #     means[i] = np.mean(results[results[:, 0] == id, 1])
# #     stds[i] = np.std(results[results[:, 0] == id, 1])
# # plt.plot(ids, means, 'k.-')
# # plt.errorbar(ids, means, yerr=stds, c='k', capsize=5, elinewidth=1)

plt.figure(figsize=(8, 6))
data = np.array([(100, 0.4271000027656555), (200, 0.638400018215179), (300, 0.7049999833106995), (400, 0.7555999755859375), (500, 0.7903000116348267),(600, 0.7962999939918518), (700, 0.8083000183105469),   (800, 0.1009000018290213), (900, 0.14800000190734863),(1000, 0.10279999673366547)])
data2 = np.array([(800, 0.8184000253677368), (900, 0.8392000198364258), (1000, 0.8379999995231628)])
plt.plot(data[:, 0], data[:, 1], 'o-', c='#ca5895')
plt.plot(data2[:, 0], data2[:, 1], 'ko-')
plt.legend(['Trainable P lr=0.001','Trainable P lr=0.0001'], loc=3)

# plt.ylim([0, 1])
plt.xlim(0, 1025)
plt.ylabel('Accuracy')
plt.xlabel('Intrinsic dimension')
# plt.legend(['Linear', 'Power', 'RFF'], loc=4)
plt.grid()

# plt.show()
plt.savefig('plots/trainable_p_breakdown.pdf')

# # power w/ different coefficients
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
# plt.figure(figsize=(8, 6))
# for lambda_s, lambda_c in experiments:
#     rows = (results[:, 0] == lambda_s) & (results[:, 1] == lambda_c)
#     ids = np.argsort(results[rows, 2])
#     accuracies = results[rows, 3][ids]
#     plt.plot(results[rows, 2][ids], accuracies, '.-')
# plt.ylabel('Accuracy')
# plt.xlabel('Intrinsic dimension')
# plt.legend([f'$\lambda_s: {lambda_s}; \lambda_c: {lambda_c}$' for lambda_s, lambda_c in experiments],
#            fontsize='x-small', loc=4)
# plt.ylim([0.3, 0.9])
# plt.xlim([0, 1025])
# plt.grid()
# # plt.show()
#
# plt.savefig('plots/power_coefficients.pdf')