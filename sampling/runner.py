import os
import numpy as np
import torch
import random
import h5py

from sampling.mh import train
from sampling.sample_objective import SampleObjective
from sampling.pyro_sample import mcmc_samples
from sampling.siren_utils import get_siren_model
from sampling.utils import save_numpy_arrays

class Runner:
    def __init__(self, conf):
        self.conf = conf

        self.func = getattr(self.conf, 'func')
        self.func_type = getattr(self.conf, 'func_type')
        self.dims = getattr(self.conf, 'dims', 2)
        self.epochs = getattr(self.conf, 'epochs', 1000)
        self.warm_up = getattr(self.conf, 'warm_up', 50)
        self.mcmc_type = getattr(self.conf, 'mcmc_type')

        self.temp = getattr(self.conf, 'temp', 1.0)
        self.use_jacobian = getattr(self.conf, 'use_jacobian', True)
        self.use_bounding_box = getattr(self.conf, 'use_bounding_box', False)
        self.reject_outside_bounds = getattr(self.conf, 'reject_outside_bounds', False)
        self.manual_jacobian = getattr(self.conf, 'manual_jacobian', None)

        self.min_coord, self.max_coord = getattr(self.conf, 'coord_lims', (-2., 2.))
        self.num_grid = getattr(self.conf, 'num_grid', 200)
        self.nonzero_dims = getattr(self.conf, 'nonzero_dims', (0,1))

        self.filename = getattr(self.conf, 'filename')
        


    def run(self):
        seed = getattr(self.conf, 'seed', None)

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if self.func_type == 'siren_model':
            model = get_siren_model(self.func)
            def func(x):
                x = x.cuda()
                return model(x).cpu()
            self.func = func

        results = {}
            
        function = SampleObjective(func=self.func, temp=self.temp, dim_x=self.dims, use_jacobian=self.use_jacobian, use_bounding_box=self.use_bounding_box, manual_jacobian=self.manual_jacobian)

        init_x = np.random.uniform(-1, 1, size=[self.dims])
        if self.mcmc_type == 'mh':
            overall_xs, acceptance_prob = train(init_x=init_x, function=function, epochs=self.epochs, warm_up=self.warm_up, reject_outside_bounds=self.reject_outside_bounds)
            results['acceptance_prob'] = acceptance_prob
        else:
            overall_xs, diagnostics = mcmc_samples(input_function=function, start_value=torch.tensor(init_x ), mcmc_type=self.mcmc_type, num_samples=self.epochs, warmup_steps=self.warm_up, need_cuda=False) 

            for name, diagnostic_dict in diagnostics.items():
                for sub_name, diagnostic in diagnostic_dict.items():
                    results['diagnostic_' + name + '_' + sub_name] =diagnostic

        results['overall_xs'] = overall_xs

        f_data = [self.func(torch.unsqueeze(torch.tensor(x).float(),dim=0)).numpy() for x in overall_xs]
        results['f_data'] = f_data

        coords = np.linspace(self.min_coord, self.max_coord, self.num_grid)
        x_coords, y_coords = np.meshgrid(coords, coords)
        rows, cols = np.shape(x_coords)
        prob = np.zeros((rows, cols ))
        gradient = np.zeros((rows, cols, self.dims))
        f_values = np.zeros((rows, cols))
        log_det_jacob = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                value = np.zeros(self.dims)
                value[self.nonzero_dims[0]] = x_coords[i][j]
                value[self.nonzero_dims[1]] = y_coords[i][j]
                value = np.array(value)

                prob[i, j] = np.exp(function.log_p(value))
                gradient[i,j, :] = function.u_gradient_fn(value)
                log_det_jacob[i,j] = function._log_det_jacobian(torch.tensor(np.array([value]), requires_grad=True).float())
                f_values[i,j] = self.func(torch.tensor(np.array([value])).float())

        results['prob'] = prob
        results['x_coords'] = x_coords
        results['y_coords'] = y_coords
        results['gradient'] = gradient
        results['f_values'] = f_values
        results['log_dit_jacobian'] = log_det_jacob
        

        save_numpy_arrays(self.filename, results)



