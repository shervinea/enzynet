"""Download PDB files."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from absl import app
from absl import flags
from absl import logging

from tqdm import tqdm
from typing import Text

import joblib
import multiprocessing
import os
import urllib.error
import urllib.request

FLAGS = flags.FLAGS

_RCSB_DOWNLOAD_WEBSITE = 'https://files.rcsb.org/download/'

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(CURRENT_DIRECTORY, '../')
PDB_PATH = os.path.join(CURRENT_DIRECTORY, '../../files/PDB/')

flags.DEFINE_integer('max_threads', lower_bound=1, upper_bound=4, default=1,
                     help='Number of threads requesting PDB downloads in '
                     'parallel. This number should be kept at a low value '
                     '(typically below 4) to avoid sending the RCSB server too '
                     'many requests at a time.')


def get_effective_download_threads(requested_threads: int) -> int:
    n_threads = min(requested_threads, multiprocessing.cpu_count())
    logging.info(f'Using {n_threads} thread(s).')
    return n_threads


def _download_pdb_file(pdb_id: Text) -> None:
    """Download a single PDB file of a given ID."""
    local_pdb_file = os.path.join(PDB_PATH, f'{pdb_id.lower()}.pdb')
    remote_pdb_file = f'{_RCSB_DOWNLOAD_WEBSITE}{pdb_id.lower()}.pdb'
    if not os.path.isfile(local_pdb_file):
        try:
            urllib.request.urlretrieve(remote_pdb_file, local_pdb_file)
        except urllib.error.HTTPError as error:
            logging.warning(f'Could not download PDB ID: {pdb_id}. '
                            f'Reason: {error}')


def download_pdb_files(n_threads: int) -> None:
    """Downloads all PDB files to the package's dedicated folder."""
    with open(os.path.join(DATASETS_PATH, 'download_pdb.txt')) as file:
        pdb_ids = file.readlines()[0].split(', ')
    joblib.Parallel(n_jobs=n_threads)(
        joblib.delayed(_download_pdb_file)(pdb_id) for pdb_id in tqdm(pdb_ids))


def main(_):
    n_threads = get_effective_download_threads(FLAGS.max_threads)
    download_pdb_files(n_threads)


if __name__ == '__main__':
    app.run(main)
