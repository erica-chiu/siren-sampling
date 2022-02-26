import math
import numpy as np
import h5py

def check_sdf(func, min_value, max_value, num_steps):
    expanded_values = np.linspace(min_value,max_value, num_steps)
    x_coords, y_coords,z_coords = np.meshgrid(expanded_values, expanded_values, expanded_values)
    negative_values = []
    nonnegative_values = []
    rows,cols,height = x_coords.shape
    for i in range(rows):
        for j in range(cols):
            for k in range(height):
                coord = np.array([[x_coords[i][j][k], y_coords[i][j][k], z_coords[i][j][k]]])
                value = func(coord)
                if value < 0:
                    negative_values.append(coord)
                else:
                    nonnegative_values.append(coord)
    return negative_values, nonnegative_values

    
def chebishev(xs, n):
    xs_squared = np.power(xs, 2)
    total = 0.
    for k in range(n // 2 + 1):
        total += math.comb(n, 2*k) * np.power(xs_squared - 1, k) * np.power(xs, n - 2 * k)
    return total
n=4
def heart(x):
    xs = x[:, 0]
    ys = x[:,1]
    zs = x[:,2]
    return chebishev(xs, n) + chebishev(ys, n) + chebishev(zs, n) 

def heart_actual(x):
    xs = x[:, 0]
    ys = x[:,1]
    zs = x[:,2]
    x_squared = np.power(xs,2)
    y_squared = np.power(ys,2)
    z_squared = np.power(zs,2)
    z_cubed = np.power(zs,3)
    return  np.power(x_squared + 9/4 * y_squared + z_squared - 1, 3) - x_squared * z_cubed - 9/80 * y_squared * z_cubed

negative_values, nonnegative_values = check_sdf(heart_actual, -2., 2., 50)
with h5py.File('/mnt/ejchiu/siren-sampling/check_sdf/heart.hdf5', 'w') as f:
    f.create_dataset('negative_values', data=negative_values)
    f.create_dataset('nonnegative_values', data=nonnegative_values)
