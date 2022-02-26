import numpy as np
import cvxpy as cp

def min_convex_comb(evaluated_functions):
    """
    :param evaluated_functions: [n, m] matrix where there are m functions and n dimensions of x
    :return:
    """
    n, m = np.shape(evaluated_functions)
    if m == 2:
        return convex_comb2(evaluated_functions[:,0], evaluated_functions[:,1])
    return general_convex_comb(evaluated_functions)[0]

def convex_comb2(v1, v2):
    if np.dot(v1-v2, v2)  >= 0 or np.dot(v2-v1, v1)  >= 0:
        if np.linalg.norm(v1) < np.linalg.norm(v2):
            return np.array([1,0])
        else:
            return np.array([0,1])
    else:
        diff_vec = v1 - v2
        denominator = np.sum((v1-v2)**2)
        return np.array([np.abs(np.dot(diff_vec, v2)/denominator), np.abs(np.dot(diff_vec, v1)/denominator)])

def general_convex_comb(evaluated_functions):
    num_dim, num_funcs = np.shape(evaluated_functions)
    x = cp.Variable([num_funcs, 1])
    cost = cp.sum_squares(evaluated_functions @ x)

    # sum to 1 constraint
    ones = np.ones([1, num_funcs])

    prob = cp.Problem(cp.Minimize(cost), [ones @ x == 1, x >= 0])
    prob.solve()
    return x.value.T


def leapfrog_int(momentum, weights, mass_inv, function, num_steps, total_time):
    """
    :param momentum: [1, n]
    :param weights: [1, n]
    :param mass_inv: [n,n]
    :param function:
    :param num_steps: scalar
    :param total_time: scalar
    :return:
    """
    dt = total_time / num_steps
    total_weights = [weights]
    for _ in range(num_steps):
        momentum = momentum - dt/2*function.u_gradient_fn(weights)
        weights = weights + dt*np.matmul(mass_inv, momentum)
        momentum = momentum - dt/2*function.u_gradient_fn(weights)
        total_weights.append(weights)
    return weights, momentum, np.array(total_weights)


def get_acceptance_prob(new_weights, new_momentum, old_weights, old_momentum, function, mass_inv):
    """
    :param new_weights: [1, n]
    :param new_momentum: [1, n]
    :param old_weights: [1, n]
    :param old_momentum: [1, n]
    :param function:
    :param mass_inv: [n, n]
    :return:
    """
    log_neg_prob = function.get_alpha(new_weights, new_momentum, mass_inv)  - function.get_alpha(old_weights, old_momentum, mass_inv)
    return np.exp(-log_neg_prob)


def train(x, function, epochs=1000, integration_steps=100, momentum_sigma=1., mass_size=1., warm_up=50, total_time=1.):
    size = np.shape(x)[0]
    momentum_cov = momentum_sigma*np.eye(size)
    mass_inv = 1/mass_size*np.eye(size)
    overall_x = []
    integration_x = []
    int_momentum = None
    num_accepted = 0
    for i in range(epochs):
        overall_x.append(x)

        # generate momentum (per weight)
        momentum = np.random.multivariate_normal(np.zeros(size), momentum_cov)
        #momentum = np.zeros(size, dtype=float)

        # leapfrog integration via Hamiltonian equations
        new_x, new_momentum, total_weights = leapfrog_int(momentum, x, mass_inv, function, num_steps=integration_steps, total_time=total_time)
        if len(integration_x) == 0:
            integration_x = total_weights
            int_momentum = momentum

        # MH acceptance step
        if np.random.uniform() < get_acceptance_prob(new_x, new_momentum, x, momentum, function, mass_inv):
            x = new_x
            num_accepted += 1
    print("acceptance rate: ", num_accepted/epochs)
    return np.array(overall_x[warm_up:]), (integration_x, int_momentum)


