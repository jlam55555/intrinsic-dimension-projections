import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import glob
import numpy as np

#set fonts
font = {'family': 'cmr10', 'size': 18}
mpl.rc('font', **font)  # change the default font to Computer Modern Roman
mpl.rcParams['axes.unicode_minus'] = False  # because cmr10 does not have a Unicode minus sign

file_handle2 = open('runs/dists.out', 'w')

figsize = (12, 8)
plt.figure(figsize=figsize)

files = glob.glob('runs/*.pkl')
for filename in files:
    with open(filename, 'rb') as file_handle:
        model_dict = pickle.load(file_handle)

    intrinsic_dim = model_dict['intrinsic_dim']

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
                # l = int(weight.shape[1]/3)
                # after_flat1.append(weight[:,0:l].flatten())
                # after_flat2.append(weight[:,l:2*l].flatten())
                # after_flat3.append(weight[:,2*l:].flatten())
                l = int(weight.shape[0]/3)
                after_flat1.append(weight[0:l,:].flatten())
                after_flat2.append(weight[l:2*l,:].flatten())
                after_flat3.append(weight[2*l:,:].flatten())

    after_flat = np.concatenate(after_flat).flatten()
    after_flat1 = np.concatenate(after_flat1).flatten()
    after_flat2 = np.concatenate(after_flat2).flatten()
    after_flat3 = np.concatenate(after_flat3).flatten()

    file_handle2.write(f'{intrinsic_dim} {before_flat.std()} {after_flat.std()} {after_flat1.std()} {after_flat2.std()} {after_flat3.std()} {intrinsic_weights_before.std()} {intrinsic_weights_after.std()}')
    file_handle2.flush()
    print(intrinsic_dim, before_flat.std(), after_flat.std(), after_flat1.std(), after_flat2.std(), after_flat3.std())
    print(intrinsic_weights_before.std(), intrinsic_weights_after.std())

    if intrinsic_dim not in (100, 500, 1000):
        continue

    # plt.subplot(1, 5, 1)
    # plt.hist(after_flat, bins=np.linspace(-0.5, 0.5, 50))
    # plt.xlabel('$\\theta_d$')
    # plt.ylabel('Frequency')
    # plt.grid()
    # plt.savefig(f'plots/mnist_{intrinsic_dim}_P_before.pdf')

    row = [100,500,1000].index(intrinsic_dim)

    plt.subplot(3, 4, 1+4*row)
    plt.hist(after_flat1, bins=np.linspace(-0.5, 0.5, 50))
    plt.xlabel('Linear weights')
    plt.ylabel('Frequency')
    # plt.grid()
    # plt.savefig(f'plots/mnist_{intrinsic_dim}_P_lin.pdf')

    plt.subplot(3, 4, 2+4*row)
    plt.hist(after_flat2, bins=np.linspace(-0.5, 0.5, 50))
    plt.xlabel('Squared weights')
    plt.ylabel('Frequency')
    # plt.grid()
    # plt.savefig(f'plots/mnist_{intrinsic_dim}_P_squ.pdf')

    plt.subplot(3, 4, 3+4*row)
    plt.hist(after_flat3, bins=np.linspace(-0.5, 0.5, 50))
    plt.xlabel('Cubed weights')
    plt.ylabel('Frequency')
    # plt.grid()
    # plt.savefig(f'plots/mnist_{intrinsic_dim}_P_cub.pdf')

    # plt.subplot(1, 6, 5)
    # plt.hist(intrinsic_weights_before.flatten(), bins=np.linspace(-0.25, 0.25, 50))
    # plt.xlabel('$\\theta_d$')
    # plt.ylabel('Frequency')
    # plt.grid()
    # plt.savefig(f'plots/mnist_{intrinsic_dim}_int_before.pdf')

    plt.subplot(3, 4, 4+4*row)
    plt.hist(intrinsic_weights_after.flatten(), bins=np.linspace(-0.025, 0.025, 50))
    plt.xlabel('$\\theta_d$')
    plt.ylabel('Frequency')
    # plt.grid()
    # plt.savefig(f'plots/mnist_{intrinsic_dim}_int_after.pdf')

plt.tight_layout()
plt.savefig(f'plots/mnist_pdist.pdf')

file_handle2.close()