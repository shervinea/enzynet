"""Volume generation and augmentation."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import keras
import os.path

import numpy as np

from enzynet import constants
from enzynet import pdb
from sklearn import decomposition
from tqdm import tqdm

_LENGTH_3_CUBE_FILLED_WITH_0_BUT_CENTER_1 = np.pad(np.ones((1, 1, 1)), 1,
                                                   'constant')

class VolumeDataGenerator(keras.utils.Sequence):
    """Generates batches of volumes containing 3D enzymes as well as their associated class labels on the fly.

    To be passed as argument in the fit_generator function of Keras.

    Parameters
    ----------
    v_size : int (optional, default is 32)
        Size of each side of the output volumes.

    flips : tuple of floats (optional, default is (0.2, 0.2, 0.2))
        Probabilities that the volumes are flipped respectively with respect
        to x, y, and z.

    batch_size : int (optional, default is 32)
        Number of samples in output array of each iteration of the 'generate'
        method.

    directory_precomputed : string (optional, default points to 'files/precomputed')
        Path of the precomputed files.

    directory_pdb : string (optional, default points to 'files/PDB')
        Path of the PDB files.

    labels : dict
        Dictionary linking PDB IDs to their labels.

    list_enzymes : list of strings
        List of enzymes to generate.

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
        List of weights (among the values ['charge', 'hydropathy',
        'isoelectric']) to consider as additional channels.

    scaling_weights : boolean (optional, default is True)
        If True, divides all weights by the weight that is maximum in absolute
        value.

    Example
    -------
    >>> from enzynet import volume
    >>> from enzynet import tools

    >>> labels = tools.read_dict('../datasets/dataset_single.csv',
                                 value_type=constants.ValueType.INT)
    >>> partition_red = tools.read_dict(
            '../../datasets/partition_single_red.csv',
            value_type=constants.ValueType.LIST_STRING)

    >>> generator = volume.VolumeDataGenerator(
            partition_red['train'], labels, v_size=64, flips=(0.2, 0.2, 0.2),
            batch_size=32, shuffle=True, p=5, max_radius=40,
            noise_treatment=False, weights=[], scaling_weights=True)
    """
    def __init__(
            self,
            list_enzymes: List[Text],
            labels: Dict[Text, int],
            v_size: int = 32,
            flips: Tuple[float, float, float] = (0.2, 0.2, 0.2),
            batch_size: int = 32,
            directory_precomputed: Text = constants.PRECOMPUTED_DIR,
            directory_pdb: Text = constants.PDB_DIR,
            shuffle: bool = True,
            p: int = 5,
            max_radius: float = 40,
            noise_treatment: bool = False,
            weights: List[Text] = [],
            scaling_weights: bool = True
    ) -> None:
        """Initialization."""
        self.batch_size = batch_size
        self.directory_precomputed = directory_precomputed
        self.directory_pdb = directory_pdb
        self.flips = flips
        self.labels = labels
        self.list_enzymes = list_enzymes
        self.max_radius = max_radius
        self.noise_treatment = noise_treatment
        self.n_channels = max(1, len(weights))
        self.p = p
        self.scaling_weights = scaling_weights
        self.shuffle = shuffle
        self.v_size = v_size
        self.weights = weights
        self.on_epoch_end()

    def check_precomputed(self) -> None:
        """Checks if all coordinates and weights have been precomputed, and precomputes them otherwise."""
        # Initialization.
        list_enzymes = list(self.labels)
        counter = 0

        # Loop over all enzymes.
        for pdb_id in tqdm(list_enzymes):
            # Find name of paths.
            names = [precomputed_name(pdb_id, self.directory_precomputed, 'coords', self.p)] + \
                    [precomputed_name(pdb_id, self.directory_precomputed, 'weights', self.p,
                                      weight, self.scaling_weights)
                     for weight in self.weights]

            # Precomputes all files.
            if all([os.path.isfile(name) for name in names]):  # Checks if all already exist.
                pass
            else:  # Precomputes files otherwise.
                save_coords_weights(pdb_id, self.weights, self.p, self.scaling_weights,
                                    self.directory_pdb, self.directory_precomputed)
                counter += 1
        print("Had to compute files of {0} enzymes.".format(counter))

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_enzymes))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.list_enzymes) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        # Generate indexes of the batch.
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs.
        list_enzymes_temp = [self.list_enzymes[k] for k in indexes]

        # Generate data.
        X, y = self.__data_augmentation(list_enzymes_temp)

        return X, y

    def __data_augmentation(
            self,
            list_enzymes_temp: List[Text]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns augmented data with batch_size enzymes."""  # X : (n_samples, v_size, v_size, v_size, n_channels).
        # Initialization.
        X = np.empty((self.batch_size,  # n_enzymes.
                      self.v_size,  # dimension w.r.t. x.
                      self.v_size,  # dimension w.r.t. y.
                      self.v_size,  # dimension w.r.t. z.
                      self.n_channels))  # n_channels.
        y = np.empty((self.batch_size), dtype=int)

        # Computations.
        for i in range(self.batch_size):
            # Store class.
            y[i] = self.labels[list_enzymes_temp[i]]

            # Load precomputed coordinates.
            coords = load_coords(list_enzymes_temp[i], self.p, self.directory_precomputed)
            coords = coords_center_to_zero(coords)
            coords = adjust_size(coords, v_size=self.v_size, max_radius=self.max_radius)

            # Get weights.
            local_weights = []
            for weight in self.weights:
                local_weight = load_weights(list_enzymes_temp[i], weight, self.p, self.scaling_weights,
                                            self.directory_precomputed)  # Computes extended weights.
                local_weights += [local_weight]  # Store.

            # PCA.
            coords = decomposition.PCA(n_components=3).fit_transform(coords)

            # Do flip.
            coords_temp = flip_around_axis(coords, axis=self.flips)

            if len(self.weights) == 0:
                # Convert to volume and store.
                X[i, :, :, :, 0] = coords_to_volume(coords_temp, self.v_size,
                                                    noise_treatment=self.noise_treatment)

            else:
                # Compute to weights of volume and store.
                for k in range(self.n_channels):
                    X[i, :, :, :, k] = weights_to_volume(coords_temp, local_weights[k],
                                                         self.v_size, noise_treatment=self.noise_treatment)

        return X, sparsify(y)


def coords_to_volume(coords: np.ndarray, v_size: int,
                     noise_treatment: bool = False) -> np.ndarray:
    """Converts coordinates to binary voxels."""  # Input is centered on [0,0,0].
    return weights_to_volume(coords=coords, weights=1, v_size=v_size, noise_treatment=noise_treatment)


def weights_to_volume(coords: np.ndarray, weights: Union[Sequence, int],
                      v_size: int, noise_treatment: bool = False) -> np.ndarray:
    """Converts coordinates to voxels with weights."""  # Input is centered on [0,0,0].
    # Initialization.
    volume = np.zeros((v_size, v_size, v_size))

    # Translate center.
    coords = coords + np.full((coords.shape[0], 3), (v_size-1)/2)

    # Round components.
    coords = coords.astype(int)

    # Filter rows with values that are out of the grid.
    mask = ((coords >= 0) & (coords < v_size)).all(axis=1)

    # Convert to volume.
    volume[tuple(coords[mask].T)] = weights[mask] if type(weights) != int else weights

    # Remove noise.
    if noise_treatment is True:
        volume = remove_noise(coords, volume)

    return volume


def coords_center_to_zero(coords: np.ndarray) -> np.ndarray:
    """Centering coordinates on [0,0,0]."""
    barycenter = get_barycenter(coords)
    return coords - np.full((coords.shape[0], 3), barycenter)


def adjust_size(coords: np.ndarray, v_size: int = 32,
                max_radius: float = 40) -> np.ndarray:
    return np.multiply((v_size/2-1)/max_radius, coords)


def sparsify(y: np.ndarray) -> np.ndarray:
    """Returns labels in binary NumPy array."""
    n_classes = 6
    return np.array([[1 if y[i] == j+1 else 0 for j in range(n_classes)]
                      for i in range(y.shape[0])])


def flip_around_axis(
        coords: np.ndarray,
        axis: Tuple[float, float, float] = (0.2, 0.2, 0.2)
) -> np.ndarray:
    """Flips coordinates randomly w.r.t. each axis with its associated probability."""
    for col in range(3):
        if np.random.binomial(1, axis[col]):
            coords[:, col] = np.negative(coords[:, col])
    return coords


def get_barycenter(coords: np.ndarray) -> np.ndarray:
    """Gets barycenter point of a Nx3 matrix."""
    return np.array([np.mean(coords, axis=0)])


def _is_point_in_volume_interior(x: int, y: int, z: int, v_size: int) -> bool:
    """Returns True if the point is not on the volume's frontier."""
    for dim in [x, y, z]:
        if dim in [0, v_size-1]:
            return False
    return True


def _is_point_isolated(volume: np.ndarray, x: int, y: int,
                       z: int) -> bool:
    """Returns True if there is no other point in a small cube neighborhood."""
    return np.array_equal(
        volume[x-1:x+2, y-1:y+2, z-1:z+2],
        _LENGTH_3_CUBE_FILLED_WITH_0_BUT_CENTER_1 * volume[x, y, z])


def _remove_isolated_point(volume: np.ndarray, x: int, y: int, z: int) -> None:
    """Removes point of coordinates (x, y, z) from the volume."""
    volume[x, y, z] = 0


def remove_noise(coords: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Removes isolated atoms from voxel structure."""
    # Parameters.
    v_size = volume.shape[0]

    # Computations.
    for i in range(coords.shape[0]):
        if _is_point_in_volume_interior(*coords[i,], v_size):
            if _is_point_isolated(volume, *coords[i,]):
                _remove_isolated_point(volume, *coords[i,])

    return volume


def precomputed_name(pdb_id: Text, path: Text, type_file: Text, desired_p: int,
                     weights_name: Optional[Text] = None,
                     scaling: bool = True) -> Text:
    """Returns path in string of precomputed file."""
    if type_file == 'coords':
        return os.path.join(path, f'{pdb_id.lower()}_coords_p{desired_p}.npy')
    elif type_file == 'weights':
        return os.path.join(path, f'{pdb_id.lower()}_{weights_name}_p{desired_p}_scaling{scaling}.npy')


def save_coords_weights(pdb_id: Text, list_weights: List[Text], desired_p: int,
                        scaling_weights: bool, source_path: Text,
                        dest_path: Text) -> None:
    """Computes coordinates and weights and saves them into .npy files."""
    # Initialize local PDB.
    local_PDB = pdb.PDBBackbone(pdb_id=pdb_id, path=source_path)

    # Coordinates.
    local_PDB.get_coords_extended(p=desired_p)  # Compute.
    coords = local_PDB.backbone_coords_ext  # Store.
    np.save(precomputed_name(pdb_id, dest_path, 'coords', desired_p), coords)  # Save.

    # Weights.
    for weights_name in list_weights:
        local_PDB.get_weights_extended(desired_p, weight_type=weights_name,
                                       scaling=scaling_weights)  # Compute.
        weights = local_PDB.backbone_weights_ext  # Store.
        np.save(precomputed_name(pdb_id, dest_path, 'weights', desired_p, weights_name, scaling_weights),
                weights)  # Save.


def load_coords(pdb_id: Text, desired_p: int, source_path: Text) -> np.ndarray:
    """Loads precomputed coordinates."""
    return np.load(precomputed_name(pdb_id, source_path, 'coords', desired_p))


def load_weights(pdb_id: Text, weights_name: Text, desired_p: int,
                 scaling: bool, source_path: Text) -> np.ndarray:
    """Loads precomputed weights."""
    return np.load(precomputed_name(pdb_id, source_path, 'weights', desired_p, weights_name, scaling))
