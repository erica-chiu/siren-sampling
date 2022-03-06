import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np
import torch
import math

from sampling.runner import Runner

class Config:
    # Config classes should not create any tensor, but only formulas for
    # creating them.
    def __init__(self, temp=1.0, use_jacobian=True):
        self.seed = 1003

        # Init x parameters
        self.dims = 2
        self.init_x_sigma = 0.1

        def heart(x):
            xs = x[:, 0]
            ys = x[:,1]
            x_squared = torch.pow(xs,2)
            y_squared = torch.pow(ys,2)
            y_cubed = torch.pow(ys,3)
            return  torch.pow( x_squared + y_squared - 1, 3) -   x_squared * y_cubed

        def heart_jacobian(x):
            xs = x[:,0]
            ys = x[:,1]
            x2 = torch.pow(xs,2)
            y2 = torch.pow(ys, 2)
            y3 = torch.pow(ys, 3)
            x2_y2_1 = x2 + y2 - 1
            dx = 6 * xs * torch.pow(x2_y2_1, 2) - 2 * xs * y3
            dy = 6 * ys * torch.pow(x2_y2_1, 2) - 3 * x2 * y2
            return torch.tensor([[dx, dy]])

        self.func = heart 


        self.temp = temp
        self.use_jacobian = use_jacobian 
        self.epochs = 10000
        self.warm_up = 500
        self.use_bounding_box = False 
        self.reject_outside_bounds = False 
        self.mcmc_type = 'nuts'
        self.func_type = 'function'
        self.manual_jacobian = heart_jacobian

        # Training parameters
        # self.momentum_sigma = 0.005
        # self.total_time = 1.0
        # self.integration_steps = 100
        # self.mass_size = 1.0

        # Saving paths

        self.filename = '/mnt/ejchiu/siren-sampling/logs/heart_2d/sampling/'
        # if self.inexact:
        #     self.filename += 'inexact_'
        # if self.normalize:
        #     self.filename += 'norm_'
        self.filename += 'temp_' + str(self.temp) + '_'
        if self.use_jacobian:
            self.filename += "jacobian_"
        if self.use_bounding_box:
            self.filename += "bounded_"
        if self.reject_outside_bounds:
            self.filename += "reject_"
        if self.manual_jacobian:
            self.filename += "manual_"
        self.filename += self.mcmc_type 
        self.filename += '_result.hdf5'



if __name__ == '__main__':

    for temp in [0.0001, 0.001, 0.01, 0.1, 0.00001, 1.]:
        for use_jacobian in [True, False ]:
            conf = Config(temp=temp, use_jacobian=use_jacobian)
            print(conf.temp)
            runner = Runner(conf)
            runner.run()
    print("done")
