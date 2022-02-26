import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import h5py


# data = np.genfromtxt('pointclouds/a_1.0b_2.0c_3.0ellipse_points.xyz')
data = h5py.File('results/temp_1.0_jacobian_result.hdf5', 'r')['overall_xs']
x = data[:,0]
y = data[:,1]
z = data[:, 2]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.show()
