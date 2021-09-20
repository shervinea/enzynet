"""Testing file for volume.py."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from typing import Text

import unittest

import numpy as np

from enzynet import volume
from parameterized import parameterized

_N_DIMENSIONS = 3


class RemoveNoiseTest(unittest.TestCase):

    @parameterized.expand([
        ['empty_volume', np.array([]), 9],
        ['frontier_point', np.array([[0, 0, 0]]), 5],
        ['adjacent_points', np.array([[0, 0, 0], [1, 1, 1]]), 3],
    ])
    def test_no_removal(self, name: Text, coords: np.ndarray,
                        v_size: int) -> None:
        shape = (v_size,) * _N_DIMENSIONS
        input_volume = np.zeros(shape)
        for coord in coords:
            input_volume[tuple(coord)] = 1
        output_volume = np.copy(input_volume)
        self.assertTrue(
            np.array_equal(output_volume,
                           volume.remove_noise(coords, input_volume)))

    @parameterized.expand([
        ['isolated_point', np.array([[1, 1, 1]]), 3],
        ['multiple_isolated_points', np.array([[1, 1, 1], [3, 3, 3]]), 5],
    ])
    def test_removal(self, name: Text, coords: np.ndarray, v_size: int) -> None:
        shape = (v_size,) * _N_DIMENSIONS
        input_volume = np.zeros(shape)
        for coord in coords:
            input_volume[tuple(coord)] = 1
        output_volume = np.zeros(shape)
        self.assertTrue(
            np.array_equal(output_volume,
                           volume.remove_noise(coords, input_volume)))


if __name__ == '__main__':
    unittest.main()
