'Generate coordinates and weights stored in .npy for faster computations'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from enzynet import VolumeDataGenerator, read_dict


# Load enzymes
enzymes = read_dict('../../datasets/dataset_single.csv')

# Generate p = 0
print("Generation for p = {0}".format(0))
volume_generator = VolumeDataGenerator(p = 0, weights = ['hydropathy', 'charge', 'isoelectric'],
                                       scaling_weights = True)
volume_generator.check_precomputed(enzymes)

# Generate p = 5
print("Generation for p = {0}".format(5))
volume_generator = VolumeDataGenerator(p = 5, weights = ['hydropathy', 'charge', 'isoelectric'],
                                       scaling_weights = True)
volume_generator.check_precomputed(enzymes)
