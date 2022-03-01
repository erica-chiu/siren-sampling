import numpy as np
import meshplot as mp
import h5py

mp.offline()
data = h5py.File('/Users/ericachiu/Documents/MEng/siren-sampling/uniformity/temp_0.0001_jacobian_bounded_mh_result.hdf5')
v, f, sample_density, triangle_areas = data['v'], data['f'], data['sample_density'], data['triangle_area']
v, f, sample_density, triangle_areas = np.array(v), np.array(f), np.array(sample_density), np.array(triangle_areas)

mp.plot(v, f )
