import os
import numpy as np
import torch
import random
import h5py

import modules
from sampling import train
from sample_objective import SampleObjective

class SampleRunner:
    def __init__(self, conf):
        self.conf = conf

        self.model_name = getattr(self.conf, 'model_name')
        self.dims = getattr(self.conf, 'dims', 2)
        self.epochs = getattr(self.conf, 'epochs', 1000)
        self.warm_up = getattr(self.conf, 'warm_up', 50)

        self.temp = getattr(self.conf, 'temp', 1.0)
        self.use_jacobian = getattr(self.conf, 'use_jacobian', True)

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
        model.eval()
        function = SampleObjective(model=model, temp=self.temp, dim_x=self.dims, use_jacobian=self.use_jacobian)

        init_x = np.random.uniform(-1, 1, size=[self.dims])
        overall_xs = train(init_x=init_x, function=function, epochs=self.epochs, warm_up=self.warm_up)
        f_data = [model(torch.unsqueeze(torch.tensor(x).cuda().float(),dim=0)).detach().cpu().numpy() for x in overall_xs]
        xs = overall_xs[:, 0]
        ys = overall_xs[:, 1]
        # min_x, min_y, max_x, max_y = min(xs)-1, min(ys)-1, max(xs)+2, max(ys)+2
        # x_coords, y_coords = np.meshgrid(np.linspace(min_x, max_x, 500), np.linspace(min_y,max_y, 500))
        # rows, cols = np.shape(x_coords)
        # prob = np.zeros((rows, cols))
        # for i in range(rows):
        #     for j in range(cols):
        #         prob[i, j] = np.exp(function.log_p(np.array([x_coords[i][j], y_coords[i][j]])))

        with h5py.File(self.filename, "w") as f:
            f.create_dataset('overall_xs', data=overall_xs)
            f.create_dataset('f_data', data=f_data)
            # f.create_dataset('prob', data=prob)
            # f.create_dataset('x_coords', data=x_coords)
            # f.create_dataset('y_coords', data=y_coords)


