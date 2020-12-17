import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
#set fonts
font = {'family': 'cmr10', 'size': 18}
mpl.rc('font', **font)  # change the default font to Computer Modern Roman
mpl.rcParams['axes.unicode_minus'] = False  # because cmr10 does not have a Unicode minus sign

# regular vs. power
mnist_files = glob.glob('runs/mnist_normalized_p_correct/mnist*.pkl')
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

mnist_files = glob.glob('runs/bad/mnist*.pkl')
results_div = []
for filename in mnist_files:
    with open(filename, 'rb') as file_handle:
        model_dict = pickle.load(file_handle)
    results_div.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])

print(results_linear)
print(results_power)

plt.figure(figsize=(8,6))
results = np.array(sorted(results_linear))
print(results)
plt.plot(results[:, 0], results[:, 1], 'bo-', label='Linear')
results = np.array(sorted(results_power))
#results[8][1] = 0.8362
plt.plot(results[:, 0], results[:, 1], 'ro-', label='Power')
results = np.array(sorted(results_rff))
#results[8][1] = 0.8069
plt.plot(results[:, 0], results[:, 1], 'ko-', label='RFF')
# results = np.array(sorted(results_rff))
# plt.plot(results[:, 0], results[:, 1], 'o-', label='RFF', color="#ca5a95")
#plt.title('784-200-200 FC MNIST Accuracy vs. Intrinsic Dimension')
plt.ylabel('Accuracy')
plt.xlabel('Intrinsic dimension')
plt.grid()
#plt.yticks(np.arange(0,1,0.05))
#plt.xticks(np.arange(0,1100,100))
#plt.ylim(top=1)
plt.xlim(0,1025)
plt.ylim(0.3, 0.9)
plt.legend(loc='lower left')
plt.savefig('plots/mnist_normalized_p_correct.pdf')

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
