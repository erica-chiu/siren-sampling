import numpy as np
from open3d import *

cloud = io.read_point_cloud("test.ply")
visualization.draw_geometries([cloud])