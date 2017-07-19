'32x32x32 Enzynet architecture with adapted weights'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import numpy as np

import os.path

from enzynet import read_dict, get_class_weights
from enzynet import VolumeDataGenerator, MetricsHistory, Voting

from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

from keras import backend as K


mode_run = 'test'
mode_dataset = 'full'
mode_weights = 'balanced'

##----------------------------- Parameters -----------------------------------##
# PDB
weights = []
n_classes = 6
n_channels = 1 + len(weights)
scaling_weights = True

# Volume
p = 0
v_size = 32
flips = (0.2, 0.2, 0.2)
max_radius = 40
shuffle = True
noise_treatment = False

# Model
batch_size = 32
max_epochs = 200
period_checkpoint = 50

# Testing
voting_type = 'probabilities'
augmentation = ['None', 'flips', 'weighted_flips']

# Miscellenaous
current_file_name = os.path.basename(__file__)[:-3]
stddev_conv3d = np.sqrt(2.0/(n_channels))

##------------------------------ Dataset -------------------------------------##
# Load dictionary of labels
dictionary = read_dict('../../datasets/dataset_single.csv')

# Load partitions
if mode_dataset == 'full':
    partition = read_dict('../../datasets/partition_single.csv')
elif mode_dataset == 'reduced':
    partition = read_dict('../../datasets/partition_single_red.csv')
exec("partition['train'] = " + partition['train'])
exec("partition['validation'] = " + partition['validation'])
exec("partition['test'] = " + partition['test'])

# Final computations
partition['train'] = partition['train'] + partition['validation']
partition['validation'] = partition['test']

# Get class weights
class_weights = get_class_weights(dictionary, partition['train'],
                                  mode = mode_weights)

# Check if data has been precomputed
VolumeDataGenerator(p = p, weights = weights,
                    scaling_weights = scaling_weights).check_precomputed(dictionary)

# Training generator
training_generator = \
    VolumeDataGenerator(v_size = v_size,
                        flips = flips,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        p = p,
                        max_radius = max_radius,
                        noise_treatment = noise_treatment,
                        weights = weights,
                        scaling_weights = scaling_weights).generate(dictionary,
                                                                    partition['train'])

# Validation generator
validation_generator = \
    VolumeDataGenerator(v_size = v_size,
                        flips = (0, 0, 0), # No flip
                        batch_size = batch_size,
                        shuffle = False, # Validate with fixed set
                        p = p,
                        max_radius = max_radius,
                        noise_treatment = noise_treatment,
                        weights = weights,
                        scaling_weights = scaling_weights).generate(dictionary,
                                                                    partition['validation'])

##----------------------------- Testing --------------------------------------##
# Voting object
predictions = \
    Voting(voting_type = voting_type,
           v_size = v_size,
           augmentation = augmentation,
           p = p,
           max_radius = max_radius,
           noise_treatment = noise_treatment,
           weights = weights,
           scaling_weights = scaling_weights)

##------------------------------ Model ---------------------------------------##
# Create
model = Sequential()

# Add layers
model.add(Conv3D(filters = 32,
                 kernel_size = 9,
                 strides = 2,
                 padding = 'valid',
                 kernel_initializer = RandomNormal(mean = 0.0, stddev = stddev_conv3d * 9**(-3/2)),
                 bias_initializer = 'zeros',
                 kernel_regularizer = l2(0.001),
                 bias_regularizer = None,
                 input_shape = (v_size, v_size, v_size, n_channels)))

model.add(LeakyReLU(alpha = 0.1))

model.add(Dropout(rate = 0.2))

model.add(Conv3D(filters = 64,
                 kernel_size = 5,
                 strides = 1,
                 padding = 'valid',
                 kernel_initializer = RandomNormal(mean = 0.0, stddev = stddev_conv3d * 5**(-3/2)),
                 bias_initializer = 'zeros',
                 kernel_regularizer = l2(0.001),
                 bias_regularizer = None))

model.add(LeakyReLU(alpha = 0.1))

model.add(MaxPooling3D(pool_size = (2,2,2)))

model.add(Dropout(rate = 0.3))

model.add(Flatten())

model.add(Dense(units = 128,
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.01),
                bias_initializer = 'zeros',
                kernel_regularizer = l2(0.001),
                bias_regularizer = None))

model.add(Dropout(rate = 0.4))

model.add(Dense(units = n_classes,
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.01),
                bias_initializer = 'zeros',
                kernel_regularizer = l2(0.001),
                bias_regularizer = None))

model.add(Activation('softmax'))

# Track accuracy and loss in real-time
history = MetricsHistory(saving_path = current_file_name + '.csv')

# Checkpoints
checkpoints = ModelCheckpoint('checkpoints/' + current_file_name + '_{epoch:02d}' + '.hd5f',
                              save_weights_only = True,
                              period = period_checkpoint)

if mode_run == 'train':
    # Compile
    model.compile(optimizer = Adam(lr = 0.001, decay = 0.00016667),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    # Train
    model.fit_generator(generator = training_generator,
                        steps_per_epoch = len(partition['train'])//batch_size,
                        epochs = max_epochs,
                        verbose = 1,
                        validation_data = validation_generator,
                        validation_steps = len(partition['validation'])//batch_size,
                        callbacks = [history, checkpoints],
                        class_weight = class_weights,
                        workers = 4)

if mode_run == 'test':
    # Load weights
    weights_path = \
        'checkpoints/' + current_file_name + '_{0:02d}'.format(max_epochs-1) + '.hd5f'
    model.load_weights(weights_path)

# Predict
predictions.predict(model, partition['test'], dictionary)

# Compute indicators
predictions.get_assessment()

# Clear session
K.clear_session()
