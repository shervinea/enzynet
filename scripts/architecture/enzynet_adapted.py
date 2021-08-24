"""32x32x32 Enzynet architecture with adapted weights."""

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

MODE_RUN = 'test'
MODE_DATASET = 'full'
MODE_WEIGHTS = 'balanced'

##----------------------------- Parameters -----------------------------------##
# PDB.
WEIGHTS = []
N_CLASSES = 6
N_CHANNELS = 1 + len(WEIGHTS)
SCALING_WEIGHTS = True

# Volume.
P = 0
V_SIZE = 32
FLIPS = (0.2, 0.2, 0.2)
MAX_RADIUS = 40
SHUFFLE = True
NOISE_TREATMENT = False

# Model.
BATCH_SIZE = 32
MAX_EPOCHS = 200
PERIOD_CHECKPOINT = 50

# Testing.
VOTING_TYPE = 'probabilities'
AUGMENTATION = ['None', 'flips', 'weighted_flips']

# Miscellaneous.
CURRENT_FILE_NAME = os.path.basename(__file__)[:-3]
STDDEV_CONV3D = np.sqrt(2.0/N_CHANNELS)

##------------------------------ Dataset -------------------------------------##
# Load dictionary of labels.
DICTIONARY = tools.read_dict('../../datasets/dataset_single.csv')

# Load partitions.
if MODE_DATASET == 'full':
    partition = tools.read_dict('../../datasets/partition_single.csv')
elif MODE_DATASET == 'reduced':
    partition = tools.read_dict('../../datasets/partition_single_red.csv')
exec("partition['train'] = " + partition['train'])
exec("partition['validation'] = " + partition['validation'])
exec("partition['test'] = " + partition['test'])

# Final computations.
partition['train'] = partition['train'] + partition['validation']
partition['validation'] = partition['test']

# Get class weights.
class_weights = tools.get_class_weights(DICTIONARY, partition['train'],
                                        mode=MODE_WEIGHTS)

# Training generator.
training_generator = volume.VolumeDataGenerator(
    list_enzymes=partition['train'],
    labels=DICTIONARY,
    v_size=V_SIZE,
    flips=FLIPS,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    p=P,
    max_radius=MAX_RADIUS,
    noise_treatment=NOISE_TREATMENT,
    weights=WEIGHTS,
    scaling_weights=SCALING_WEIGHTS)

# Validation generator.
validation_generator = volume.VolumeDataGenerator(
    list_enzymes=partition['validation'],
    labels=DICTIONARY,
    v_size=V_SIZE,
    flips=(0, 0, 0),  # No flip.
    batch_size=BATCH_SIZE,
    shuffle=False,  # Validate with fixed set.
    p=P,
    max_radius=MAX_RADIUS,
    noise_treatment=NOISE_TREATMENT,
    weights=WEIGHTS,
    scaling_weights=SCALING_WEIGHTS)

# Check if data has been precomputed.
training_generator.check_precomputed()

##----------------------------- Testing --------------------------------------##
# Voting object.
predictions = keras_utils.Voting(
    list_enzymes=partition['test'],
    labels=DICTIONARY,
    voting_type=VOTING_TYPE,
    v_size=V_SIZE,
    augmentation=AUGMENTATION,
    p=P,
    max_radius=MAX_RADIUS,
    noise_treatment=NOISE_TREATMENT,
    weights=WEIGHTS,
    scaling_weights=SCALING_WEIGHTS)

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
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=STDDEV_CONV3D * 9 ** (-3 / 2)),
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=None,
        input_shape=(V_SIZE, V_SIZE, V_SIZE, N_CHANNELS)))

model.add(advanced_activations.LeakyReLU(alpha=0.1))

model.add(layers.Dropout(rate=0.2))

model.add(
    layers.Conv3D(
        filters=64,
        kernel_size=5,
        strides=1,
        padding='valid',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=STDDEV_CONV3D * 5 ** (-3 / 2)),
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
        units=N_CLASSES,
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=None))

model.add(layers.Activation('softmax'))

# Track accuracy and loss in real-time.
history = keras_utils.MetricsHistory(saving_path=CURRENT_FILE_NAME + '.csv')

# Checkpoints.
checkpoints = callbacks.ModelCheckpoint(
    'checkpoints/' + CURRENT_FILE_NAME + '_{epoch:02d}' + '.hd5f',
    save_weights_only=True,
    period=PERIOD_CHECKPOINT)

if MODE_RUN == 'train':
    # Compile.
    model.compile(optimizer=optimizers.Adam(lr=0.001, decay=0.00016667),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train.
    model.fit_generator(generator=training_generator,
                        epochs=MAX_EPOCHS,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=[history, checkpoints],
                        class_weight=class_weights,
                        use_multiprocessing=True,
                        workers=6,
                        max_queue_size=30)

if MODE_RUN == 'test':
    # Load weights.
    weights_path = \
        'checkpoints/' + CURRENT_FILE_NAME + '_{0:02d}'.format(MAX_EPOCHS) + '.hd5f'
    model.load_weights(weights_path)

# Predict.
predictions.predict(model)

# Compute indicators.
predictions.get_assessment()

# Clear session.
K.clear_session()
