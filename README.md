# Simple-NN-BP-FP

Extract Behler-Parrinello fingerprint (https://doi.org/10.1063/1.3553717) calculation module written in C++ from SIMPLE-NN package (https://github.com/MDIL-SNU/SIMPLE-NN). Access this fingerprint calculation function using python.

# Compile

```
pip install mpi4py
git clone https://github.com/yilinyang1/Simple-NN-BP-FP.git
cd Simple-NN-BP-FP/utils
python libsymf_builder.py
```

# Usage

```python
from utils.fp_calculator import set_sym, calculate_fp

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
                     
# calculate fingerprint
fp_data = calculate_fp(atoms, elements, params_set, is_mpi=False)
fp = fp_data['x']
dfp = fp_data['dx]
```
