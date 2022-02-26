import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import h5py


# data = np.genfromtxt('pointclouds/a_1.0b_2.0c_3.0ellipse_points.xyz')
data = h5py.File('results/temp_1.0_jacobian_result.hdf5', 'r')
x_coords = data['x_coords']
y_coords = data['y_coords']
z_coords = data['z_coords']
gradients = data['gradient']
gradients = np.array(gradients) * 0.001
density = data['prob']

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.quiver(x_coords, y_coords, z_coords, gradients[:,:, :, 0], gradients[:,:, :,1], gradients[:,:, :,2])
half = len(density)//2
print(z_coords[:, half, :])
plt.contourf(y_coords[:, half, :], z_coords[:, half, :], density[:,half,:])
plt.show()
