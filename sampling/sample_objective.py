import torch
import numpy as np



class SampleObjective:
    def __init__(self, func, temp, dim_x, use_jacobian=True, use_bounding_box=False, manual_jacobian=None):
        self.func = func 
        self.temp = temp
        #self.normalize = normalize
        self.use_jacobian = use_jacobian
        self.use_bounding_box = use_bounding_box
        self.dim_x = dim_x
        self.manual_jacobian = manual_jacobian


    def _barrier(self, x):
        boundary_value = 1.
        outside = torch.logical_or(x > boundary_value, x < - boundary_value).float()
        result = (torch.pow(10*(torch.abs(x) - boundary_value), 10)) * outside 
        return torch.sum(result)


    def _norm_f(self, x):
        """
        ||f(x)||^2/T
        :param first_derivatives: [d, num_f]
        :param coefs: [num_f]
        :return:
        """
        result = self.func(x)
        if self.use_bounding_box:
            result += self._barrier(x)
        norm = torch.sum(torch.pow(result, 2))
        return norm / self.temp

    def _log_det_jacobian(self, x):
        jacobian = None
        if self.manual_jacobian:
            jacobian = self.manual_jacobian(x)
        else:
            func_out = self.func(x)
            if self.use_bounding_box:
                func_out += self._barrier(x)
            jacobian = torch.autograd.grad(func_out, x, grad_outputs=torch.ones_like(func_out), create_graph=True)[0]
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
            x = x.float()
            
        # x = x.clone().detach()
        if not x.requires_grad:
            x = x.clone().detach()
            x.requires_grad_()
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
        return self.u_fn(weights).detach().numpy() + momentum.T @ mass_inv @ momentum

    def u_gradient_fn(self, x):
        x = torch.tensor(x, requires_grad=True).float()
        result = self.u_fn(x, not_tensor=False)
        return torch.autograd.grad(result, x, grad_outputs=torch.ones_like(result), create_graph=True)[0].detach().numpy()

    def log_p(self, x, not_tensor=True):
        return -self.u_fn(x, not_tensor).detach().numpy()


