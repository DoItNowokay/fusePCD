# import libraries
import os
import numpy as np
from IPython.display import Image as ipy_image
from IPython.display import display
from termcolor import colored, cprint

# import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = False  # Disable benchmark for CPU

# import model creation and utilities
from models.base_model import create_model
from utils.util_3d import render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif

# options for the model. please check `utils/demo_util.py` for more details
from utils.demo_util import SDFusionText2ShapeOpt

# Set random seed
seed = 2023
# Initialize options for CPU
opt = SDFusionText2ShapeOpt(gpu_ids=None, seed=seed)  # Set gpu_ids to None for CPU

# Initialize SDFusion model
ckpt_path = 'saved_ckpt/sdfusion-txt2shape.pth'
opt.init_model_args(ckpt_path=ckpt_path)

# Create the model
SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')

# txt2shape
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)

# Change the input text here to generate different shapes!
input_txt = "A rocking chair"

ngen = 6  # Number of generated shapes
ddim_steps = 100
ddim_eta = 0.
uc_scale = 3.

sdf_gen = SDFusion.txt2shape(input_txt=input_txt, ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)

mesh_gen = sdf_to_mesh(sdf_gen)

# # Visualize as GIF
# gen_name = f'{out_dir}/txt2shape-{input_txt}.gif'
# save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)

# print(f'Input: "{input_txt}"')
# display(ipy_image(gen_name))
