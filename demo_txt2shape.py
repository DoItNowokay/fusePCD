# first set up which gpu to use
import os
gpu_ids = 1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

# import libraries
import numpy as np
from IPython.display import Image as ipy_image
from IPython.display import display
from termcolor import colored, cprint

# import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
# import torchvision.utils as vutils

from models.base_model import create_model
from utils.util_3d import render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif


# options for the model. please check `utils/demo_util.py` for more details
from utils.demo_util import SDFusionText2ShapeOpt

seed = 2023
opt = SDFusionText2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device
# device = 'cpu'

# initialize SDFusion model
ckpt_path = 'saved_ckpt/sdfusion-txt2shape.pth'
opt.init_model_args(ckpt_path=ckpt_path)

SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')


# txt2shape
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)

# change the input text here to generate different chairs/tables!
input_txt = "Chair on Table"

ngen = 6 # number of generated shapes
ddim_steps = 100
ddim_eta = 0.
uc_scale = 3.

sdf_gen = SDFusion.txt2shape(input_txt=input_txt, ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)


mesh_gen = sdf_to_mesh(sdf_gen)
# pc_gen = sdf_to_point_cloud(sdf_gen, level=0.02, render_all=True)
# # print(mesh_gen.shape)
# # print(type(mesh_gen))
# import pickle
# with open(f'outputs/{input_txt}.pkl', 'wb') as f:
#     pickle.dump(pc_gen, f)
# # vis as gif
# print(sdf_gen.shape)
gen_name = f'{out_dir}/txt2shape-{input_txt}.gif'
save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)

# print(f'Input: "{input_txt}"')
# for name in [gen_name]:
#     display(ipy_image(name))
