import numpy as np
import torch
import dataio
import os


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_sdf_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    slice_coords_2d = dataio.get_mgrid(512)

    with torch.no_grad():
        yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                     torch.zeros_like(slice_coords_2d[:, :1]),
                                     slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     -0.75*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

        min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
        min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)


