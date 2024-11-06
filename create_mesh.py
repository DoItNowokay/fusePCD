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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{0}"

# options for the model. please check `utils/demo_util.py` for more details
from utils.demo_util import SDFusionText2ShapeOpt
import pickle

seed = 2023
opt = SDFusionText2ShapeOpt(gpu_ids=0, seed=seed)
device = opt.device
# device = 'cpu'

# initialize SDFusion model
ckpt_path = 'saved_ckpt/sdfusion-txt2shape.pth'
opt.init_model_args(ckpt_path=ckpt_path)

SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')

with open('output.pkl', 'rb') as f:
    sdf_gen = pickle.load(f)


out_dir = 'demo_results'

mesh_gen = sdf_to_mesh(sdf_gen)

input_txt = "A chair with no armrests."    
gen_name = f'{out_dir}/txt2shape-{input_txt}.gif'
save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)