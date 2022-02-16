import os
import numpy as np
import torch
import random
import h5py

import modules
from sampling import train
from sample_objective import SampleObjective
import diff_operators

class SampleDensityRunner:
    def __init__(self, conf):
        self.conf = conf

        self.model_name = getattr(self.conf, 'model_name')
        self.dims = getattr(self.conf, 'dims', 2)
        self.epochs = getattr(self.conf, 'epochs', 1000)
        self.warm_up = getattr(self.conf, 'warm_up', 50)

        self.temp = getattr(self.conf, 'temp', 1.0)
        self.use_jacobian = getattr(self.conf, 'use_jacobian', True)
        self.use_bounding_box = getattr(self.conf, 'use_bounding_box', False)
        self.reject_outside_bounds = getattr(self.conf, 'reject_outside_bounds', False)

        self.filename = getattr(self.conf, 'filename')


    def run(self):
        seed = getattr(self.conf, 'seed', None)

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        class SDFDecoder(torch.nn.Module):
            def __init__(self, model_name):
                super().__init__()
                # Define the model.
                self.model = modules.SingleBVPNet(type='sine', final_layer_factor=1, in_features=3)
                self.model.load_state_dict(torch.load(model_name))
                self.model.cuda()

            def forward(self, coords):
                model_in = {'coords': coords}
                return self.model(model_in)['model_out']


        model = SDFDecoder(self.model_name)
        # model.eval()
        function = SampleObjective(model=model, temp=self.temp, dim_x=self.dims, use_jacobian=self.use_jacobian, use_bounding_box=self.use_bounding_box)

        densities = []
        xs = np.linspace(self.min_x, self.max_x, self.num_xs)
        ys = []
        for x in xs:
            y = self.corresponding_y(x)
            ys.append(y)
            total = 0
            samples = self.num_samples**2
            start_x, end_x = x - self.grid_length/2, x + self.grid_length/2
            start_y, end_y = y - self.grid_length/2, y + self.grid_length/2
            for sample_x in np.linspace(start_x, end_x, self.num_samples):
                for sample_y in np.linspace(start_y, end_y, self.num_samples):
                    total += np.exp(function.log_p(np.array([sample_x, sample_y])))
            densities.append(total / samples / self.get_arc_length(start_x, end_x, start_y, end_y))
        

        num_grid = 50
        min_coord, max_coord = -1.1, 1.1
        x_coords, y_coords, z_coords = np.meshgrid(np.linspace(min_coord, max_coord, num_grid), np.linspace(min_coord, max_coord, num_grid), np.linspace(min_coord, max_coord, num_grid))
        rows, cols, height = np.shape(x_coords)
        prob = np.zeros((rows, cols, height))
        gradient = np.zeros((rows, cols, height, 3))
        f_values = np.zeros((rows, cols, height))
        for i in range(rows):
            for j in range(cols):
                for k in range(height):
                    value = np.array([x_coords[i][j][k], y_coords[i][j][k], z_coords[i][j][k]])
                    prob[i, j, k] = np.exp(function.log_p(value))
                    gradient[i,j,k, :] = function.u_gradient_fn(value)
                    f_values[i,j,k] = model(torch.tensor(np.array([value])).cuda().float())


        with h5py.File(self.filename, "w") as f:
            f.create_dataset('overall_xs', data=overall_xs)
            f.create_dataset('f_data', data=f_data)
            f.create_dataset('prob', data=prob)
            f.create_dataset('x_coords', data=x_coords)
            f.create_dataset('y_coords', data=y_coords)
            f.create_dataset('z_coords', data=z_coords)
            f.create_dataset('gradient', data=gradient)
            f.create_dataset('f_values', data=f_values)
            f.create_dataset('acceptance_prob', data=acceptance_prob)


