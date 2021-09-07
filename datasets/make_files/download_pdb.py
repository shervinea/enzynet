"""Converts list of PDB enzymes in a text file."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import pandas as pd


def create_download_file() -> None:
    """Creates the RCSB download file containing all relevant PDB IDs."""
    df = pd.read_csv('../dataset_all.csv', header=None)
    with open('../download_pdb.txt', 'w') as file:
        file.write(", ".join(df[0].tolist()))


if __name__ == "__main__":
    create_download_file()
