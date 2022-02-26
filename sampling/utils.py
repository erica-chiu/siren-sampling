import numpy as np





        path = os.path.dirname(os.path.abspath(self.filename))
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(self.filename, "w") as f:
            f.create_dataset('overall_xs', data=overall_xs)
            f.create_dataset('f_data', data=f_data)
            f.create_dataset('prob', data=prob)
            f.create_dataset('x_coords', data=x_coords)
            f.create_dataset('y_coords', data=y_coords)
            f.create_dataset('z_coords', data=z_coords)
            # f.create_dataset('gradient', data=gradient)
            f.create_dataset('f_values', data=f_values)
            # f.create_dataset('acceptance_prob', data=acceptance_prob)
            if diagnostics:
                for name, diagnostic_dict in diagnostics.items():
                    for sub_name, diagnostic in diagnostic_dict.items():
                        f.create_dataset('diagnostic_' + name + '_' + sub_name, data=diagnostic)
