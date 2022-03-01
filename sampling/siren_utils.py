import numpy as np
import torch

def get_siren_model(model_name):

    class SDFDecoder(torch.nn.Module):
        def __init__(self, model_name):
            super().__init__()
            # Define the model.
            self.model = modules.SingleBVPNet(type='sine', final_layer_factor=1, in_features=3)
            self.model.load_state_dict(torch.load(model_name))
            self.model.cuda()

        def forward(self, coords):
            model_in = {'coords': coords}
            return self.model.forward_with_grad(model_in)['model_out']


    model = SDFDecoder(self.func)
    return model
