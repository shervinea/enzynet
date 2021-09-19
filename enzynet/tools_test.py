"""Testing file for tools.py."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import os
import unittest

from enzynet import constants
from enzynet import tools
from parameterized import parameterized


class TestReadDict(unittest.TestCase):

    @parameterized.expand([
        ['dataset_all', constants.DATASETS_DIR, constants.ValueType.LIST_INT],
        ['dataset_multi', constants.DATASETS_DIR, constants.ValueType.LIST_INT],
        ['dataset_single', constants.DATASETS_DIR, constants.ValueType.INT],
        ['ligands_to_pdb', constants.DATASETS_DIR,
         constants.ValueType.LIST_STRING],
        ['partition_single', constants.DATASETS_DIR,
         constants.ValueType.LIST_STRING],
        ['partition_single_red', constants.DATASETS_DIR,
         constants.ValueType.LIST_STRING],
        ['pdb_to_ligands', constants.DATASETS_DIR,
         constants.ValueType.LIST_STRING],
        ['enzynet_adapted', constants.ARCHITECTURE_DIR,
         constants.ValueType.LIST_FLOAT],
        ['enzynet_uniform', constants.ARCHITECTURE_DIR,
         constants.ValueType.LIST_FLOAT],
    ])
    def test_successful_read(self, name, root_dir, value_type):
        tools.read_dict(os.path.join(root_dir, f'{name}.csv'), value_type)


if __name__ == '__main__':
    unittest.main()
