import numpy as np
import torch
import math

from sample_runner import SampleRunner

class Config:
    # Config classes should not create any tensor, but only formulas for
    # creating them.
    def __init__(self, temp=1.0):
        self.seed = 1000

        # Init x parameters
        self.dims = 2
        self.init_x_sigma = 0.1

        self.experiment_path = '/mnt/ejchiu/siren-sampling/logs/' + 'experiment_2'

        self.model_name = self.experiment_path +'/models/model_9999.pth'
        self.temp = temp
        self.use_jacobian = False
        self.epochs = 1000
        self.warm_up = 50

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
        self.filename += 'result.hdf5'



if __name__ == '__main__':
    for temp in [1.]:
        conf = Config(temp=temp)
        print(conf.temp)
        runner = SampleRunner(conf)
        runner.run()
    print("done")
