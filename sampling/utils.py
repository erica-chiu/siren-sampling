import numpy as np
import h5py
import os


def save_numpy_arrays(filename, array_dict, create_dir_if_not_exist=True):

    if create_dir_if_not_exist:
        path = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(path):
            os.makedirs(path)

    with h5py.File(filename, "w") as f:
        for data_name, data in array_dict.items():
            f.create_dataset(data_name, data=data)
