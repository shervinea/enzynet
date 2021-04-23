"""32x32x32 Enzynet architecture with uniform weights."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import os.path

import numpy as np

from enzynet import keras_utils
from enzynet import tools
from enzynet import volume
from keras import backend as K
from keras import callbacks
from keras import initializers
from keras import layers
from keras.layers import advanced_activations
from keras import models
from keras import optimizers
from keras import regularizers

mode_run = 'test'
mode_dataset = 'full'
mode_weights = 'unbalanced'

##----------------------------- Parameters -----------------------------------##
# PDB.
weights = []
n_classes = 6
n_channels = 1 + len(weights)
scaling_weights = True

# Volume.
p = 0
v_size = 32
flips = (0.2, 0.2, 0.2)
max_radius = 40
shuffle = True
noise_treatment = False

# Model.
batch_size = 32
max_epochs = 200
period_checkpoint = 50

# Testing.
voting_type = 'probabilities'
augmentation = ['None', 'flips', 'weighted_flips']

# Miscellaneous.
current_file_name = os.path.basename(__file__)[:-3]
stddev_conv3d = np.sqrt(2.0/(n_channels))

##------------------------------ Dataset -------------------------------------##
# Load dictionary of labels.
dictionary = tools.read_dict('../../datasets/dataset_single.csv')

# Load partitions.
if mode_dataset == 'full':
    partition = tools.read_dict('../../datasets/partition_single.csv')
elif mode_dataset == 'reduced':
    partition = tools.read_dict('../../datasets/partition_single_red.csv')
exec("partition['train'] = " + partition['train'])
exec("partition['validation'] = " + partition['validation'])
exec("partition['test'] = " + partition['test'])

# Final computations.
partition['train'] = partition['train'] + partition['validation']
partition['validation'] = partition['test']

# Get class weights.
class_weights = tools.get_class_weights(dictionary, partition['train'],
                                  mode=mode_weights)

# Training generator.
training_generator = volume.VolumeDataGenerator(
    list_enzymes=partition['train'],
    labels=dictionary,
    v_size=v_size,
    flips=flips,
    batch_size=batch_size,
    shuffle=shuffle,
    p=p,
    max_radius=max_radius,
    noise_treatment=noise_treatment,
    weights=weights,
    scaling_weights=scaling_weights)

# Validation generator.
validation_generator = volume.VolumeDataGenerator(
    list_enzymes=partition['validation'],
    labels=dictionary,
    v_size=v_size,
    flips=(0, 0, 0),  # No flip.
    batch_size=batch_size,
    shuffle=False,  # Validate with fixed set.
    p=p,
    max_radius=max_radius,
    noise_treatment=noise_treatment,
    weights=weights,
    scaling_weights=scaling_weights)

# Check if data has been precomputed.
training_generator.check_precomputed()

##----------------------------- Testing --------------------------------------##
# Voting object.
predictions = keras_utils.Voting(
    list_enzymes=partition['test'],
    labels=dictionary,
    voting_type=voting_type,
    v_size=v_size,
    augmentation=augmentation,
    p=p,
    max_radius=max_radius,
    noise_treatment=noise_treatment,
    weights=weights,
    scaling_weights=scaling_weights)

##------------------------------ Model ---------------------------------------##
# Create.
model = models.Sequential()

# Add layers.
model.add(
    layers.Conv3D(
        filters=32,
        kernel_size=9,
        strides=2,
        padding='valid',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=stddev_conv3d * 9**(-3/2)),
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=None,
        input_shape=(v_size, v_size, v_size, n_channels)))

model.add(advanced_activations.LeakyReLU(alpha=0.1))

model.add(layers.Dropout(rate=0.2))

model.add(
    layers.Conv3D(
        filters=64,
        kernel_size=5,
        strides=1,
        padding='valid',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=stddev_conv3d * 5**(-3/2)),
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=None))

model.add(advanced_activations.LeakyReLU(alpha=0.1))

model.add(layers.MaxPooling3D(pool_size=(2,2,2)))

model.add(layers.Dropout(rate=0.3))

model.add(layers.Flatten())

model.add(
    layers.Dense(
        units=128,
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=None))

model.add(layers.Dropout(rate=0.4))

model.add(
    layers.Dense(
        units=n_classes,
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=None))

model.add(layers.Activation('softmax'))

# Track accuracy and loss in real-time.
history = keras_utils.MetricsHistory(saving_path=current_file_name + '.csv')

# Checkpoints.
checkpoints = callbacks.ModelCheckpoint(
    'checkpoints/' + current_file_name + '_{epoch:02d}' + '.hd5f',
    save_weights_only=True,
    period=period_checkpoint)

if mode_run == 'train':
    # Compile.
    model.compile(optimizer=optimizers.Adam(lr=0.001, decay=0.00016667),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train.
    model.fit_generator(generator=training_generator,
                        epochs=max_epochs,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=[history, checkpoints],
                        class_weight=class_weights,
                        use_multiprocessing=True,
                        workers=8,
                        max_queue_size=30)

if mode_run == 'test':
    # Load weights.
    weights_path = \
        'checkpoints/' + current_file_name + '_{0:02d}'.format(max_epochs) + '.hd5f'
    model.load_weights(weights_path)

# Predict.
predictions.predict(model)

# Compute indicators.
predictions.get_assessment()

# Clear session.
K.clear_session()
