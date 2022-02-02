import numpy as np

def get_acceptance_prob(new_x, x, function):
    return np.exp(function.log_p(new_x) - function.log_p(x))

def train(init_x, function, epochs, warm_up=50):
    size = np.shape(init_x)[0]
    x = init_x
    overall_x = []
    num_accepted = 0
    for _ in range(epochs):
        overall_x.append(x)
        # print(x)

        # sample step
        new_x = np.random.normal(x, 1.0, size=size)
        # print(get_acceptance_prob(new_x, x, function))

        # MH acceptance step
        if np.random.uniform() < get_acceptance_prob(new_x, x, function):
            x = new_x
            num_accepted += 1
    print("acceptance rate: ", num_accepted / epochs)
    return np.array(overall_x[warm_up:])