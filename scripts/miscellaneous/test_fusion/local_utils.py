'Local additional functions'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import os.path
import numpy as np

from enzynet import VolumeDataGenerator
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
precomputed_path = os.path.join(current_directory, '../../../files/precomputed/')
PDB_path = os.path.join(current_directory, '../../../files/PDB/')

n_classes = 6

class Probabilities(VolumeDataGenerator):
    """
    Predicts probabilities of testing enzymes

    """
    def __init__(self, v_size = 32, batch_size = 32, directory_precomputed = precomputed_path,
                 directory_pdb = PDB_path, shuffle = True, p = 0, max_radius = 40,
                 noise_treatment = False, weights = [], scaling_weights = True):
        'Initialization'
        VolumeDataGenerator.__init__(self, v_size = v_size, flips = (0, 0, 0), batch_size = batch_size,
                                     directory_precomputed = directory_precomputed,
                                     directory_pdb = directory_pdb, shuffle = shuffle, p = p,
                                     max_radius = max_radius, noise_treatment = noise_treatment,
                                     weights = weights, scaling_weights = scaling_weights)

    def predict(self, model, list_enzymes, dictionary):
        'Predicts classes of testing enzymes'
        # Initialization
        self.y_pred = np.empty((len(list_enzymes), n_classes))
        self.y_true = np.array([dictionary[enzyme] for enzyme in list_enzymes], dtype = int)
        self.y_id = np.array(list_enzymes)

        # Computations
        for i, enzyme in enumerate(tqdm(list_enzymes)):
            self.y_pred[i,:] = self.__vote(model, dictionary, enzyme)

        return self.y_pred

    def __vote(self, model, dictionary, enzyme):
        # Generate volume
        X = self._VolumeDataGenerator__data_augmentation(dictionary, [enzyme])[0]

        # Return probability
        return np.array(model.predict_proba(X, verbose = 0))
