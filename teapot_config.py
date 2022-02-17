import numpy as np
import torch
import math

from sample_runner import SampleRunner

class Config:
    # Config classes should not create any tensor, but only formulas for
    # creating them.
    def __init__(self, temp=1.0, use_jacobian=True):
        self.seed = 1000

        # Init x parameters
        self.dims = 3
        self.init_x_sigma = 0.1

        self.experiment_path = '/mnt/ejchiu/siren-sampling/logs/' + 'experiment_teapot_side_length'

        self.model_name = self.experiment_path +'/checkpoints/model_final.pth'
        self.temp = temp
        self.use_jacobian = use_jacobian 
        self.epochs = 10000
        self.warm_up = 500
        self.use_bounding_box = True 
        self.reject_outside_bounds = False 
        self.mcmc_type = 'hmc'

        # Training parameters
        # self.momentum_sigma = 0.005
        # self.total_time = 1.0
        # self.integration_steps = 100
        # self.mass_size = 1.0

        # Saving paths

        self.filename = self.experiment_path + '/sampling/'
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
        self.filename += self.mcmc_type + '_'
        self.filename += 'result.hdf5'



if __name__ == '__main__':
    for use_jacobian in [True, False]:
        for temp in [0.001, 0.0001, 0.01, 0.1, 0.00001, 1.]:
            conf = Config(temp=temp, use_jacobian=use_jacobian)
            print(conf.temp)
            runner = SampleRunner(conf)
            runner.run()
    print("done")
