import numpy as np
import torch
import math

from func_runner import SampleRunner

class Config:
    # Config classes should not create any tensor, but only formulas for
    # creating them.
    def __init__(self, temp=1.0, use_jacobian=True):
        self.seed = 1003

        # Init x parameters
        self.dims = 3
        self.init_x_sigma = 0.1

        def ellipsoid(x):
            a=1. / 3
            b=2. /3
            c=3. / 3
            xs = x[:, 0]
            ys = x[:,1]
            zs = x[:,2]
            return 1 - (xs/a)**2 - (ys/b)**2 - (zs/c)**2

        self.func = ellipsoid


        self.temp = temp
        self.use_jacobian = use_jacobian 
        self.epochs = 10000
        self.warm_up = 500
        self.use_bounding_box = True 
        self.reject_outside_bounds = False 
        self.mcmc_type = 'nuts'

        # Training parameters
        # self.momentum_sigma = 0.005
        # self.total_time = 1.0
        # self.integration_steps = 100
        # self.mass_size = 1.0

        # Saving paths

        self.filename = '/mnt/ejchiu/siren-sampling/logs/direct/sampling/'
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
        self.filename += self.mcmc_type 
        self.filename += 'tbd_result.hdf5'



if __name__ == '__main__':

    for temp in [0.0001, 0.001, 0.01, 0.1, 0.00001, 1.]:
        for use_jacobian in [True, False ]:
            conf = Config(temp=temp, use_jacobian=use_jacobian)
            print(conf.temp)
            runner = SampleRunner(conf)
            runner.run()
    print("done")
