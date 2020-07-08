from ._libsymf import lib, ffi
from .gen_ffi import _gen_2Darray_for_ffi
import numpy as np
from .mpiclass import DummyMPI, MPI4PY


def set_sym(elements, Gs, cutoff, g2_etas=None, g2_Rses=None, g4_etas=None, g4_zetas=None, g4_lambdas=None):
	"""
	specify symmetry function parameters for each element
	parameters for each element contain:
	integer parameters: [which sym func, surrounding element 1, surrounding element 1]
						surrouding element starts from 1. For G2 sym func, the third 
						element is 0. For G4 and G5, the order of the second and the
						third element does not matter.
	double parameters:  [cutoff radius, 3 sym func parameters]
						for G2: eta, Rs, dummy
						for G4 and G5: eta, zeta, lambda
	"""

	# specify all elements in the system
	params_set = dict()
	ratio = 36.0  # difference ratio from the AMP parameters
	for item in elements:
		params_set[item] = dict()
		int_params = []
		double_params = []
		for G in Gs:
			if G == 2:
				int_params += [[G, el1, 0] for el1 in range(1, len(elements)+1) 
										   for g2_eta in g2_etas
										   for g2_Rs in g2_Rses]
				double_params += [[cutoff, g2_eta/ratio, g2_Rs, 0] for el1 in range(1, len(elements)+1)
																   for g2_eta in g2_etas
																   for g2_Rs in g2_Rses]
			else:
				int_params += [[G, el1, el2] for el1 in range(1, len(elements)+1)
											 for el2 in range(el1, len(elements)+1)
											 for g4_eta in g4_etas
											 for g4_zeta in g4_zetas
											 for g4_lambda in g4_lambdas]
				double_params += [[cutoff, g4_eta/ratio, g4_zeta, g4_lambda] 
											 for el1 in range(1, len(elements)+1)
											 for el2 in range(el1, len(elements)+1)
											 for g4_eta in g4_etas
											 for g4_zeta in g4_zetas
											 for g4_lambda in g4_lambdas]


		params_set[item]['i'] = np.array(int_params, dtype=np.intc)
		params_set[item]['d'] = np.array(double_params, dtype=np.float64)
		params_set[item]['ip'] = _gen_2Darray_for_ffi(params_set[item]['i'], ffi, "int")
		params_set[item]['dp'] = _gen_2Darray_for_ffi(params_set[item]['d'], ffi)
		params_set[item]['total'] = np.concatenate((params_set[item]['i'], params_set[item]['d']), axis=1)
		params_set[item]['num'] = len(params_set[item]['total'])

	return params_set


def calculate_fp(atoms, elements, params_set, is_mpi=False):
	"""
		atoms: ase Atoms class
		symbols: list of unique elements in atoms
	"""
	try:
		import mpi4py
	except ImportError:
		comm = DummyMPI()
	else:
		if is_mpi:
			comm = MPI4PY()
		else:
			comm = DummyMPI()

	cart = np.copy(atoms.get_positions(wrap=True), order='C')
	scale = np.copy(atoms.get_scaled_positions(), order='C')
	cell = np.copy(atoms.cell, order='C')

	cart_p  = _gen_2Darray_for_ffi(cart, ffi)
	scale_p = _gen_2Darray_for_ffi(scale, ffi)
	cell_p  = _gen_2Darray_for_ffi(cell, ffi)

	atom_num = len(atoms.positions)
	symbols = np.array(atoms.get_chemical_symbols())
	atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
	type_num = dict()
	type_idx = dict()
	for j,jtem in enumerate(elements):
		tmp = symbols==jtem
		atom_i[tmp] = j+1
		type_num[jtem] = np.sum(tmp).astype(np.int64)
		type_idx[jtem] = np.arange(atom_num)[tmp]
	atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

	res = dict()
	res['x'] = dict()
	res['dx'] = dict()


	for j,jtem in enumerate(elements):
		q = type_num[jtem] // comm.size
		r = type_num[jtem] %  comm.size

		begin = comm.rank * q + min(comm.rank, r)
		end = begin + q

		if r > comm.rank:
			end += 1

		cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')
		cal_num = len(cal_atoms)
		cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

		x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
		dx = np.zeros([cal_num, atom_num * params_set[jtem]['num'] * 3], dtype=np.float64, order='C')

		x_p = _gen_2Darray_for_ffi(x, ffi)
		dx_p = _gen_2Darray_for_ffi(dx, ffi)
		errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
						 atom_i_p, atom_num, cal_atoms_p, cal_num, \
						 params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
						 x_p, dx_p)
		comm.barrier()
		errnos = comm.gather(errno)
		errnos = comm.bcast(errnos)

		if isinstance(errnos, int):
			errnos = [errno]

		for errno in errnos:
			if errno == 1:
				err = "Not implemented symmetry function type."
				raise NotImplementedError(err)
			elif errno == 2:
				err = "Zeta in G4/G5 must be greater or equal to 1.0."
				raise ValueError(err)
			else:
				assert errno == 0
		
		if type_num[jtem] != 0:
			res['x'][jtem] = np.array(comm.gather(x, root=0))
			res['dx'][jtem] = np.array(comm.gather(dx, root=0))
			
			if comm.rank == 0:
				res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
				res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
									reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])

		else:
			res['x'][jtem] = np.zeros([0, params_set[jtem]['num']])
			res['dx'][jtem] = np.zeros([0, params_set[jtem]['num'], atom_num, 3])


	# the fp and dfpdX are sorted by atom type right now, rearrange them to be in the order of atom index
	n_atoms = len(atoms)
	n_features = len(params_set[elements[0]]['i'])
	image_fp = np.zeros([n_atoms, n_features])
	image_dfpdX = np.zeros([n_atoms, n_features, n_atoms, 3])

	for ie in range(1, len(elements) + 1):
		el = elements[ie-1]
		image_fp[atom_i == ie, :] = res['x'][el]
		image_dfpdX[atom_i == ie, :, :, :] = res['dx'][el]
	res['x'] = image_fp
	res['dx'] = image_dfpdX

	return res