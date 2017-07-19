'Finds optimal values of fusion and tests it'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import numpy as np

from enzynet import read_dict

from tqdm import tqdm

# Load datasets
dictionary = read_dict('../../../datasets/dataset_single.csv')
partition = read_dict('../../../datasets/partition_single.csv')
exec("partition['train'] = " + partition['train'])
exec("partition['validation'] = " + partition['validation'])
exec("partition['test'] = " + partition['test'])

## ------------------------------ Validation -------------------------------- ##
# Load datasets
m_1 = np.load('results/enzynet_binary_val.npy')
m_2 = np.load('results/enzynet_hydropathy_val.npy')
m_3 = np.load('results/enzynet_isoelectric_val.npy')

# Initialization
accs = np.zeros((101,101))
y_true = np.array([dictionary[enzyme] for enzyme in partition['validation']], dtype = int)

# Loop
for a_1 in tqdm(range(101)):
    for a_2 in range(a_1, 101):
        # Combined matrix
        result = a_1 * m_1 + (a_2-a_1) * m_2 + (100-a_2) * m_3

        # Predicted values
        y_pred = np.argmax(result, axis = 1) + 1

        # Store accuracy
        accs[a_1,a_2] = 100 * np.sum(y_true == y_pred)/result.shape[0]

# Find optimal parameters
a_1_opt, a_2_opt = np.unravel_index(accs.argmax(), accs.shape)

# Print infos
print('Validation')
print('--')
print('Optimal parameters for a_1 = {0} and a_2 = {1}'.format(a_1_opt, a_2_opt))
print('None: {0:.2f} %'.format(accs[100,100]))
print('Hydropathy: {0:.2f} %'.format(accs[0,100]))
print('Isoelectric: {0:.2f} %'.format(accs[0,0]))
print('Optimized: {0:.2f} %'.format(accs[a_1_opt,a_2_opt]))
print('')

## ------------------------------ Testing ----------------------------------- ##
# Load datasets
m_1 = np.load('results/enzynet_binary_test.npy')
m_2 = np.load('results/enzynet_hydropathy_test.npy')
m_3 = np.load('results/enzynet_isoelectric_test.npy')

# Initialization
y_true = np.array([dictionary[enzyme] for enzyme in partition['test']], dtype = int)

# None
result = 100 * m_1
y_pred = np.argmax(result, axis = 1) + 1
acc_none = 100 * np.sum(y_true == y_pred)/result.shape[0]

# Hydropathy
result = 100 * m_2
y_pred = np.argmax(result, axis = 1) + 1
acc_hydr = 100 * np.sum(y_true == y_pred)/result.shape[0]

# Isoelectric
result = 100 * m_3
y_pred = np.argmax(result, axis = 1) + 1
acc_iso = 100 * np.sum(y_true == y_pred)/result.shape[0]

# Optimized
result = a_1_opt * m_1 + (a_2_opt-a_1_opt) * m_2 + (100-a_2_opt) * m_3
y_pred = np.argmax(result, axis = 1) + 1
acc_opt = 100 * np.sum(y_true == y_pred)/result.shape[0]

# Print infos
print('Testing')
print('--')
print('None: {0:.2f} %'.format(acc_none))
print('Hydropathy: {0:.2f} %'.format(acc_hydr))
print('Isoelectric: {0:.2f} %'.format(acc_iso))
print('Optimized: {0:.2f} %'.format(acc_opt))
