import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

mnist_files = glob.glob('runs/mnist*.pkl')

results1 = []
results2 = []

for filename in mnist_files:
    with open(filename, 'rb') as file_handle:
        model_dict = pickle.load(file_handle)

    if model_dict['model_type'] == 'linear':
        results1.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])
    else:
        results2.append([model_dict['intrinsic_dim'], model_dict['eval'][1]])

results = np.array(results1)
plt.scatter(results[:,0], results[:,1])
results = np.array(results2)
plt.scatter(results[:,0], results[:,1])
plt.title('784-200-200 FC MNIST Accuracy vs. Intrinsic Dimension')
plt.ylabel('Accuracy')
plt.xlabel('Intrinsic dimension')
plt.show()
