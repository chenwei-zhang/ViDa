import numpy as np
import h5py


# A = np.random.randint(100, size=(4,4))
# B = np.random.randint(100, size=(5,3,3))

# # Save to h5 file
# f1 = h5py.File("datapy2.hdf5", "w")
# dset1 = f1.create_dataset("dataset_01", (4,4), dtype='i', data=A)
# dset2 = f1.create_dataset("dataset_02", (5,3,3), dtype='i', data=B)
# dset1.attrs['scale'] = 0.01
# dset1.attrs['offset'] = 15
# f1.close()

# Load h5 file
f2 = h5py.File('datapy2.hdf5', 'r')
print list(f2.keys()),"\n"

dset1 = f2['dataset_01']
data = dset1[:]
print data

