'Storing charge information in a dictionary'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from enzynet.tools import dict_to_csv


# Initialization
charge = {}

# Info from p. 51 of The cell: a molecular approach, by Hausman et al.
charge['ALA'] = 0
charge['ARG'] = 1
charge['ASN'] = 0
charge['ASP'] = -1
charge['CYS'] = 0
charge['GLU'] = -1
charge['GLN'] = 0
charge['GLY'] = 0
charge['HIS'] = 0.1
charge['ILE'] = 0
charge['LEU'] = 0
charge['LYS'] = 1
charge['MET'] = 0
charge['PHE'] = 0
charge['PRO'] = 0
charge['SER'] = 0
charge['THR'] = 0
charge['TRP'] = 0
charge['TYR'] = 0
charge['VAL'] = 0

# Save results
dict_to_csv(charge, '../charge.csv')
