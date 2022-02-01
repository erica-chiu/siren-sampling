import numpy as np
import random
import math

points = []

a = 1.0
b = 2.0
c = 3.0

filename = 'pointclouds/'
filename += 'a_' + str(a)
filename += 'b_' + str(b)
filename += 'c_' + str(c)
filename += 'ellipse_points.xyz'

num_points = 500000
for _ in range(num_points):
    x = random.uniform(-a, a)
    y_max = b*math.sqrt(1-(x/a)**2)
    y = random.uniform(-y_max, y_max)
    z_pos = math.sqrt(y_max - (y/b)**2)
    z = random.choice([-z_pos, z_pos])
    points.append([x,y,z, 2*x/a**2, 2*y/b**2, 2*z/c**2])

points = np.array(points)
np.savetxt(filename, points)
