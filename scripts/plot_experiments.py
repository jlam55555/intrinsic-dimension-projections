import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# set fonts
font = {'family': 'cmr10', 'size': 18}
mpl.rc('font', **font)  # change the default font to Computer Modern Roman
mpl.rcParams['axes.unicode_minus'] = False  # because cmr10 does not have a Unicode minus sign

# power w/ different coefficients
mnist_files = glob.glob('runs/power_coefficients/mnist*.pkl')
results = []
for filename in mnist_files:
    with open(filename, 'rb') as file_handle:
        model_dict = pickle.load(file_handle)

    results.append([model_dict['squared_coefficient'] if 'squared_coefficient' in model_dict else 0.1,
                    model_dict['cubed_coefficient'] if 'cubed_coefficient' in model_dict else 0.01,
                    model_dict['intrinsic_dim'],
                    model_dict['eval'][1]])

results = np.array(results)
experiments = [
    (1, 1),
    (0.5, 0.25),
    (0.1, 0.01),
    (0.01, 0.001),
    (1, 0),
    (0.5, 0),
    (0.1, 0)
]
plt.figure(figsize=(8, 6))
for lambda_s, lambda_c in experiments:
    rows = (results[:, 0] == lambda_s) & (results[:, 1] == lambda_c)
    ids = np.argsort(results[rows, 2])
    accuracies = results[rows, 3][ids]
    plt.plot(results[rows, 2][ids], accuracies, '.-')
plt.ylabel('Accuracy')
plt.xlabel('Intrinsic dimension')
plt.legend([f'$\lambda_s: {lambda_s}; \lambda_c: {lambda_c}$' for lambda_s, lambda_c in experiments],
           fontsize='x-small', loc=4)
plt.ylim([0.3, 0.9])
plt.xlim([0, 1025])
plt.grid()

plt.show()
plt.savefig('plots/power_coefficients.pdf')
