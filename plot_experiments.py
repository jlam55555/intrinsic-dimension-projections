import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

# regular vs. power
mnist_files = glob.glob('runs/mnist_nonnormalized_p/mnist*.pkl') + glob.glob('runs/rff_nonnormalized_p/*.pkl')
results_linear = []   # linear dense projection
results_power = []   # power dense projection
results_rff = []
for filename in mnist_files:
    with open(filename, 'rb') as file_handle:
        model_dict = pickle.load(file_handle)

    if model_dict['model_type'] == 'linear':
        results_linear.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
    elif model_dict['model_type'] == 'power' or model_dict['model_type'] == 'squared':
        results_power.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
    else:
        results_rff.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])

print(results_linear)
print(results_power)

#plt.figure()
results = np.array(results_linear)
plt.scatter(results[:, 0], results[:, 1])
results = np.array(results_power)
plt.scatter(results[:, 0], results[:, 1])
results = np.array(results_rff)
plt.scatter(results[:, 0], results[:, 1], color='k')
plt.title('784-200-200 FC MNIST Accuracy vs. Intrinsic Dimension')
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

plt.show()
