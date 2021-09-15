"""Pckage-level constant values."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import os

# Data.
N_CLASSES = 6

# Paths.
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets/')
MAKE_FILES_DIR = os.path.join(ROOT_DIR, 'datasets/make_files/')
RAW_DIR = os.path.join(ROOT_DIR, 'datasets/make_files/raw/')
MODULE_DIR = os.path.join(ROOT_DIR, 'enzynet/')
PDB_DIR = os.path.join(ROOT_DIR, 'files/PDB/')
PRECOMPUTED_DIR = os.path.join(ROOT_DIR, 'files/precomputed/')
VISUALIZATION_DIR = os.path.join(ROOT_DIR, 'scripts/volume/')
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'scripts/architecture/checkpoints/')
