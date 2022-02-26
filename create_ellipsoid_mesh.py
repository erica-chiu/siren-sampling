import numpy as np
import math
import matplotlib.pyplot as plt

a = 1.
b = 2.
c = 3. 

vertices = [[[0., 0., -c]]]
faces = []

num_values = 30 
eta_values = np.linspace(-math.pi/2, math.pi/2, num_values)
omega_values = np.linspace(-math.pi, math.pi, num_values)
last_idx = 0
indices = [[0]]
for eta_idx in range(1, num_values-1):
    eta = eta_values[eta_idx]
    row = [] 
    for omega_idx in range(num_values-1):
        omega = omega_values[omega_idx]
        vertex = [ a * math.cos(eta) * math.cos(omega), b * math.cos(eta) * math.sin(omega), c * math.sin(eta) ]
        row.append(vertex)
    row_indices = list(range(last_idx+1, last_idx + num_values))
    indices.append(row_indices)
    last_idx += num_values - 1
    vertices.append(row)

vertices.append([[0., 0., c]])
indices.append([last_idx+1])
last_idx += 1

for row_idx in range(1, num_values-2):
    row1 = vertices[row_idx]
    row2 = vertices[row_idx+1]
    
    idx1 = indices[row_idx]
    idx2 = indices[row_idx+1]

    for vertex_idx in range(-1, num_values - 2):
        vertex1 = idx1[vertex_idx]
        vertex2 = idx2[vertex_idx+1]
        faces.append([vertex1, vertex2, idx2[vertex_idx]])
        faces.append([vertex1, vertex2, idx1[vertex_idx+1]])


for row_idx in range(-1, num_values-2):
    faces.append([0, indices[1][row_idx], indices[1][row_idx+1]])
    faces.append([last_idx, indices[-2][row_idx], indices[-2][row_idx+1]])


filename = 'meshes/ellipse.obj'
all_vertices = [vertex for row in vertices for vertex in row] 
all_vertices = np.array(all_vertices) 
faces = np.array(faces)  + 1
with open(filename, 'w') as f:
    for vertex in all_vertices:
        f.write('v ' + ' '.join([str(x) for x in vertex]) + '\n')

    for face in faces:
        f.write('f ' + ' '.join([str(x) for x in face]) + '\n')
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2])
plt.show()

