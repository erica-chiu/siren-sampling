import numpy as np

def get_acceptance_prob(new_x, x, function):
    return np.exp(function.log_p(new_x) - function.log_p(x))

def train(init_x, function, epochs, warm_up=50, reject_outside_bounds=False):
    size = np.shape(init_x)[0]
    x = init_x
    overall_x = []
    num_accepted = 0
    for i in range(epochs):
        # print(x)

        overall_x.append(x)
        # sample step
        new_x = np.random.normal(x, 0.1, size=size)
        if reject_outside_bounds and np.any(np.logical_or(new_x > 1., new_x < -1.)):
            continue
        # print(get_acceptance_prob(new_x, x, function))

        # MH acceptance step
        if np.random.uniform() < get_acceptance_prob(new_x, x, function):
            x = new_x
            num_accepted += 1
        else:
            continue
        
    acceptance_prob =  num_accepted / epochs
    return np.array(overall_x), acceptance_prob
