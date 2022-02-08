import torch
import numpy as np



class SampleObjective:
    def __init__(self, model, temp, dim_x, use_jacobian=True):
        self.model = model
        self.temp = temp
        #self.normalize = normalize
        self.use_jacobian = use_jacobian
        self.dim_x = dim_x

    def _norm_f(self, x):
        """
        ||f(x)||^2/T
        :param first_derivatives: [d, num_f]
        :param coefs: [num_f]
        :return:
        """
        norm = torch.sum(self.model(x) ** 2)
        return norm / self.temp

    def _log_det_jacobian(self, x):
        sdf = self.model.model({'coords':x})
        jacobian = torch.autograd.grad(sdf['model_out'], sdf['model_in'], grad_outputs=torch.ones_like(sdf['model_out']), create_graph=True)[0]
        return torch.log(torch.sqrt(torch.linalg.det(jacobian @ jacobian.T)))


    def u_fn(self, x, not_tensor=True):
        """
        - ln p(x)
        :param x:
        :param not_tensor:
        :return:
        """
        if not_tensor:
            x = torch.tensor(x, requires_grad=True)
            x = x.cuda().float()
        x = torch.unsqueeze(x, dim=0)
        result = self._norm_f(x)
        if self.use_jacobian:
            result -= self._log_det_jacobian(x)
        return result

    def get_alpha(self, weights, momentum, mass_inv):
        """
        :param weights: [1, n]
        :param momentum: [1, n]
        :param function:
        :param mass_inv: [n, n]
        :return: [1,1]
        """
        return self.u_fn(weights).detach().cpu().numpy() + momentum.T @ mass_inv @ momentum

    def u_gradient_fn(self, x):
        x = torch.tensor(x, requires_grad=True).cuda().float()
        result = self.u_fn(x, not_tensor=False)
        return torch.autograd.grad(result, x, grad_outputs=torch.ones_like(result), create_graph=True)[0].detach().cpu().numpy()

    def log_p(self, x, not_tensor=True):
        return -self.u_fn(x, not_tensor).detach().cpu().numpy()


