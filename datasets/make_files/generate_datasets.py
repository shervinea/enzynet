"""Generate datasets."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from typing import Any, Dict, List, Text

from absl import app
from absl import flags

import os
import more_itertools

import numpy as np
import pandas as pd

from enzynet import constants
from enzynet import pdb
from enzynet import tools
from tqdm import tqdm

FLAGS = flags.FLAGS
np.random.seed(0)  # For reproducibility.

flags.DEFINE_bool('save', False, 'Whether to save generation results.')

# Date of retrieval of raw datasets from rcsb.org: 07-03-2017.

# Parameters.
_ELEMENT_DELIMITER = ', '
_STRINGS_TO_IGNORE = ['\'', '"', '[', ']']

N_CLASSES = 6
RATIO_TRAIN = 0.8  # Split all 80:20 in train/test and train 80:20 in train/validation.
FACTOR_REDUCTION = 0.1


def find_correct_ids(save: bool = False) -> None:
    """Finds PDB IDs that contain valid coordinates."""
    # Computations.
    for pdb_class in range(1, N_CLASSES+1):
        # Initialization.
        errors = []
        enzymes = pd.read_table(
            os.path.join(constants.RAW_DIR, f'{pdb_class}.txt'),
            header=None)
        list_enzymes = enzymes.loc[:, 0].tolist()

        # Look for errors.
        for k, enzyme in enumerate(tqdm(list_enzymes)):
            try:
                # Load local enzyme.
                local_enzyme = pdb.PDBBackbone(enzyme)
                local_enzyme.get_coords()

                # Look for problem.
                if len(local_enzyme.backbone_atoms) == 0:  # Empty enzyme.
                    errors += [k]

            except:  # File does not exist, i.e. deprecated enzyme.
                errors += [k]

        # Drop errors and save.
        enzymes = enzymes.drop(enzymes.index[errors])

        if save:  # DONE 2017-03-18.
            enzymes.to_csv(
                os.path.join(constants.MAKE_FILES_DIR, f'{pdb_class}.txt'),
                header=None, index=None)


def create_pdb_to_class_mapping(save: bool = False) -> None:
    """Creates dictionary linking PDB IDs to their corresponding classes."""
    # Initialization.
    labels_all = {}
    labels_single = {}
    labels_multi = {}

    # Compute labels of all enzymes.
    for pdb_class in range(1, N_CLASSES+1):
        # Load PDB IDs for each class.
        enzymes = pd.read_table(
            os.path.join(
                constants.MAKE_FILES_DIR, f'{pdb_class}.txt'), header=None)
        enzymes = enzymes[0].tolist()

        # Add entries to dict.
        for pdb_id in enzymes:
            if pdb_id in labels_all:  # PDB_id already in dict.
                labels_all[pdb_id] += [pdb_class]
            else:  # PDB_id not yet in dict.
                labels_all[pdb_id] = [pdb_class]

    # Create single- and multi-only datasets.
    for pdb_id, pdb_classes in labels_all.items():
        if len(pdb_classes) == 1:
            labels_single[pdb_id] = more_itertools.one(pdb_classes)
        else:
            labels_multi[pdb_id] = pdb_classes

    if save:  # DONE 2017-03-18.
        tools.dict_to_csv(
            labels_all, os.path.join(constants.DATASETS_DIR, 'dataset_all.csv'))
        tools.dict_to_csv(
            labels_single,
            os.path.join(constants.DATASETS_DIR, 'dataset_single.csv'))
        tools.dict_to_csv(
            labels_multi,
            os.path.join(constants.DATASETS_DIR, 'dataset_multi.csv'))


def create_pdb_ligand_mappings(save: bool = False) -> None:
    """Creates mappings linking PDBs to ligands."""
    # Initialization.
    ligands = {}
    pdbs = {}

    # Computations.
    pdb_ids = list(
        tools.read_dict(
            os.path.join(constants.DATASETS_DIR, 'dataset_all.csv')))
    for pdb_id in tqdm(pdb_ids):
        # Get ligands.
        local_enzyme = pdb.PDBBackbone(pdb_id)
        local_enzyme.get_ligands()

        # PDB ID -> ligands mapping.
        pdbs[pdb_id] = local_enzyme.ligands

        # Ligand -> PDB IDs mapping.
        for ligand_id in local_enzyme.ligands:
            if ligand_id in ligands:  # Already in dict.
                ligands[ligand_id] += [pdb_id]
            else:  # New entry.
                ligands[ligand_id] = [pdb_id]

    if save:  # DONE 2017-03-18.
        tools.dict_to_csv(
            pdbs, os.path.join(constants.DATASETS_DIR, 'pdb_to_ligands.csv'))
        tools.dict_to_csv(
            ligands, os.path.join(constants.DATASETS_DIR, 'ligands_to_pdb.csv'))


def split_dataset_into_train_val_test(save: bool = False) -> None:
    """Split single-label into training/validation/testing sets."""
    # Random index shuffling.
    pdb_ids = list(
        tools.read_dict(
            os.path.join(constants.DATASETS_DIR, 'dataset_single.csv')))
    indexes = np.arange(len(pdb_ids))
    np.random.shuffle(indexes)

    # Train: ratio**2, val: ratio*(1-ratio), test: 1-ratio.
    # With ratio = 0.8, train/val/test have proportion 64% / 16% / 20%.
    train_ids = [pdb_ids[i] for i in indexes[0:int(RATIO_TRAIN**2 * len(pdb_ids))]]
    val_ids = [pdb_ids[i] for i in indexes[int(RATIO_TRAIN**2 * len(pdb_ids)):int(RATIO_TRAIN * len(pdb_ids))]]
    test_ids = [pdb_ids[i] for i in indexes[int(RATIO_TRAIN * len(pdb_ids)):]]

    partition = {
        'train': train_ids,
        'validation': val_ids,
        'test': test_ids,
    }

    if save:  # DONE 2017-03-18.
        tools.dict_to_csv(
            partition,
            os.path.join(constants.DATASETS_DIR, 'partition_single.csv'))


def _create_reduced_set(list_ids: List[Text]) -> List[Text]:
    """Returns a subset of proportion FACTOR_REDUCTION from the initial set."""
    indices = np.arange(len(list_ids))
    np.random.shuffle(indices)
    return [list_ids[i] for i in indices[:int(FACTOR_REDUCTION * len(indices))]]


def _format_dict_values_to_lists_of_strings(
        raw_dict: Dict[Any, Text]) -> Dict[Any, List[Text]]:
    """Turns string values of dictionary into lists of strings."""
    out_dict = {}
    for key, value in raw_dict.items():
        for string_to_ignore in _STRINGS_TO_IGNORE:
            value = value.replace(string_to_ignore, '')
        out_dict[key] = value.split(_ELEMENT_DELIMITER)
    return out_dict


def create_reduced_sets(save: bool = False) -> None:
    """Create reduced single-label train/validation/test sets."""
    # Create reduced sets.
    partition = _format_dict_values_to_lists_of_strings(
        tools.read_dict(
            os.path.join(constants.DATASETS_DIR, 'partition_single.csv')))
    partition_red = {key: _create_reduced_set(key) for key in partition}

    if save:  # DONE 2017-03-18.
        tools.dict_to_csv(
            partition_red,
            os.path.join(constants.DATASETS_DIR, 'partition_single_red.csv'))


def create_download_file(save: bool = False) -> None:
    """Creates the RCSB download file containing all relevant PDB IDs."""
    df = pd.read_csv(os.path.join(constants.DATASETS_DIR, 'dataset_all.csv'),
                     header=None)

    if save:  # DONE 2017-03-18.
        with open(os.path.join(constants.DATASETS_DIR, 'download_pdb.txt'), 'w') as file:
            file.write(", ".join(df[0].tolist()))


def main(_):
    find_correct_ids(FLAGS.save)
    create_pdb_to_class_mapping(FLAGS.save)
    create_pdb_ligand_mappings(FLAGS.save)
    split_dataset_into_train_val_test(FLAGS.save)
    create_reduced_sets(FLAGS.save)
    create_download_file(FLAGS.save)


if __name__ == '__main__':
    app.run(main)
