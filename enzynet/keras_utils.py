"""Additional functions for Keras."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import os.path
import itertools

import numpy as np

from enzynet import indicators
from enzynet import real_time
from enzynet import tools
from enzynet import volume
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
precomputed_path = os.path.join(current_directory, '../files/precomputed/')
PDB_path = os.path.join(current_directory, '../files/PDB/')

methods = ['confusion_matrix', 'accuracy',
           'precision_per_class', 'recall_per_class', 'f1_per_class',
           'macro_precision', 'macro_recall', 'macro_f1']


class Voting(volume.VolumeDataGenerator):
    """Predicts classes of testing enzymes.

    Parameters
    ----------
    voting_type : string in {'probabilities', 'classes'} (optional, default is
                  'probabilities')
        Probabilistic- or class-based voting.

    augmentation : list of strings in {'None', 'flips', 'weighted_flips'}
                   (optional, default is ['None'])
        List of data augmentation options to perform during testing phase.

    v_size : int (optional, default is 32)
        Size of each side of the output volumes.

    directory_precomputed : string (optional, default points to 'files/precomputed')
        Path of the precomputed files.

    directory_pdb : string (optional, default points to 'files/PDB')
        Path of the PDB files.

    labels : dict
        Dictionary linking PDB IDs to their labels.

    list_enzymes : list of strings
        List of enzymes.

    shuffle : boolean (optional, default is True)
        If True, shuffles order of exploration.

    p : int (optional, default is 5)
        Interpolation of enzymes with p added coordinates between each pair
        of consecutive atoms.

    max_radius : float (optional, default is 40)
        Maximum radius of sphere that will completely fit into the volume.

    noise_treatment : boolean (optional, default is False)
        If True, voxels with no direct neighbor will be deleted.

    weights : list of strings (optional, default is [])
        List of weights (among the values ['hydropathy', 'charge']) to consider
        as additional channels.

    scaling_weights : boolean (optional, default is True)
        If True, divides all weights by the weight that is maximum in absolute
        value.
    """
    def __init__(self, list_enzymes, labels, voting_type='probabilities', augmentation=['None'],
                 v_size=32, directory_precomputed=precomputed_path,
                 directory_pdb=PDB_path, shuffle=True, p=5, max_radius=40,
                 noise_treatment=False, weights=[], scaling_weights=True):
        """Initialization."""
        volume.VolumeDataGenerator.__init__(self, list_enzymes, labels, v_size=v_size, flips=(0, 0, 0),
                                     batch_size=1, directory_precomputed=directory_precomputed,
                                     directory_pdb=directory_pdb, shuffle=shuffle, p=p,
                                     max_radius=max_radius, noise_treatment=noise_treatment,
                                     weights=weights, scaling_weights=scaling_weights)
        self.voting_type = voting_type
        self.augmentation = augmentation

    def predict(self, model):
        """Predicts classes of testing enzymes."""
        # Initialization.
        self.y_pred = np.empty((len(self.list_enzymes), len(self.augmentation)), dtype=int)
        self.y_true = np.array([self.labels[enzyme] for enzyme in self.list_enzymes], dtype=int)
        self.y_id = np.array(self.list_enzymes)

        # Computations.
        for j, augmentation in enumerate(self.augmentation):
            print('Augmentation: {0}'.format(augmentation))
            for i, enzyme in enumerate(tqdm(self.list_enzymes)):
                self.y_pred[i,j] = \
                    self.__vote(model, enzyme, augmentation)

    def get_assessment(self):
        """Compute several performance indicators."""
        for j, augmentation in enumerate(self.augmentation):
            print('Augmentation: {0}'.format(augmentation))
            ind = indicators.Indicators(self.y_true, self.y_pred[:,j])
            for method in methods:
                getattr(ind, method)()

    def __vote(self, model, enzyme, augmentation):
        # Initialization.
        probability = np.zeros((1, 6))

        # Nothing.
        if augmentation == 'None':
            # Store configuration.
            self.flips = (0, 0, 0)

            # Generate volume.
            X = self._VolumeDataGenerator__data_augmentation([enzyme])[0]

            # Voting by adding probabilities.
            if self.voting_type == 'probabilities':
                probability += model.predict_proba(X, verbose=0)
            elif self.voting_type == 'classes':
                probability[0, model.predict_classes(X, verbose=0)[0]] += 1

        # Flips.
        elif augmentation == 'flips':
            # Generate all possibilities.
            generator_flips = itertools.product(range(2), repeat=3)

            # Computations.
            for flip in generator_flips:
                # Store configuration.
                self.flips = flip

                # Generate volume.
                X = self._VolumeDataGenerator__data_augmentation([enzyme])[0]

                # Voting by adding probabilities.
                if self.voting_type == 'probabilities':
                    probability += model.predict_proba(X, verbose=0)
                elif self.voting_type == 'classes':
                    probability[0, model.predict_classes(X, verbose=0)[0]] += 1

        # Weighted flips.
        elif augmentation == 'weighted_flips':
            # Generate all possibilities.
            generator_flips = itertools.product(range(2), repeat=3)

            # Computations.
            for flip in generator_flips:
                # Store configuration.
                self.flips = flip
                factor = 1/(sum(flip)+1)

                # Generate volume.
                X = self._VolumeDataGenerator__data_augmentation([enzyme])[0]

                # Voting by adding probabilities.
                if self.voting_type == 'probabilities':
                    probability += factor * model.predict_proba(X, verbose=0)
                elif self.voting_type == 'classes':
                    probability[0, model.predict_classes(X, verbose=0)[0]] += factor * 1

        # Predict label.
        output_label = np.argmax(probability[0,:]) + 1

        return output_label


class MetricsHistory(Callback):
    """Tracks accuracy and loss in real-time, and plots it.

    Parameters
    ----------
    saving_path : string (optional, default is 'test.csv')
        Full path to output csv file.
    """
    def __init__(self, saving_path='test.csv'):
        # Initialization.
        self.display = real_time.RealTimePlot(max_entries=200)
        self.saving_path = saving_path
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
        # Store.
        self.epochs += [epoch]
        self.losses += [logs.get('loss')]
        self.val_losses += [logs.get('val_loss')]
        self.accs += [logs.get('acc')]
        self.val_accs += [logs.get('val_acc')]

        # Add point to plot.
        self.display.add(x=epoch,
                         y_tr=logs.get('acc'),
                         y_val=logs.get('val_acc'))
        plt.pause(0.001)


        # Save to file.
        dictionary = {'epochs': self.epochs,
                      'losses': self.losses,
                      'val_losses': self.val_losses,
                      'accs': self.accs,
                      'val_accs': self.val_accs}
        tools.dict_to_csv(dictionary, self.saving_path)
