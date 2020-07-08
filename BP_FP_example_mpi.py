from ase.build import fcc111
from time import time
from utils.fp_calculator import set_sym, calculate_fp
import numpy as np

# set up random AuPd slab
atoms = fcc111('Au', size=(10, 10, 10), a=4.1, vacuum=10.0)

np.random.seed(1024)
for i in set(np.random.randint(0, len(atoms), size=1000)):
    atoms[i].symbol = 'Pd' 

# set up BP symm func parameters
elements = ['Pd', 'Au']
Gs = [2, 4]
cutoff = 6.0
g2_etas = [0.05, 4.0, 20.0, 80.0]
g2_Rses = [0.0]
g4_etas = [0.005]
g4_zetas = [1.0, 4.0]
g4_lambdas = [-1.0, 1.0]

params_set = set_sym(elements, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses, 
                     g4_etas=g4_etas, g4_zetas=g4_zetas, g4_lambdas=g4_lambdas)

# calculate BP FP
t1 = time()
fp_data = calculate_fp(atoms, elements, params_set, is_mpi=True)
t2 = time()

print(t2 - t1)
print(fp_data['x'].shape)
print(fp_data['dx'].shape)

