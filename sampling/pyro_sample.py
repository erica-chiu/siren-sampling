import torch
from pyro.infer import NUTS, MCMC, HMC


def mcmc_samples(input_function, num_samples, start_value, mcmc_type, need_cuda=True, warmup_steps=200):
    if need_cuda:
        start_value = start_value.cuda()
    start_value = start_value.float()

    def potential_fn(z):
        z = z['points']
        return input_function.u_fn(z, not_tensor=False) 

    mcmc_kernel = None
    if mcmc_type == 'nuts':
        mcmc_kernel = NUTS(potential_fn=potential_fn)
    elif mcmc_type == 'hmc':
        mcmc_kernel = HMC(potential_fn=potential_fn)
    elif mcmc_type == 'mala':
        mcmc_kernel = HMC(potential_fn=potential_fn, num_steps=1)

    mcmc = MCMC(kernel=mcmc_kernel, warmup_steps=warmup_steps, initial_params={'points': start_value}, num_samples=num_samples)

    mcmc.run()
    diagnostics = mcmc.diagnostics()
    mcmc.summary()

    return mcmc.get_samples()['points'].cpu().detach().numpy(), diagnostics
