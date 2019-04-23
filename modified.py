# This is a custom scripted modified to output the inference of an individual image

import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from margipose.models.margipose_model import MargiPoseModelFactory
from margipose.models.margipose_model import HeatmapColumn, MargiPoseModel
from margipose.data_specs import ImageSpecs
from pose3d_utils.coords import ensure_homogeneous


from margipose.models import load_model
from margipose.data.skeleton import CanonicalSkeletonDesc, VNect_Common_Skeleton
from margipose.data import make_dataloader, make_unbatched_dataloader, PoseDataset

from margipose.data.get_dataset import get_dataset
from margipose.utils import seed_all, init_algorithms


CPU = torch.device('cpu')
model = load_model('./pretrained/margipose-mpi3d.pth')
model.eval()

with torch.no_grad():
    img = Image.open('./img_000001.jpg')
    # img = img.load()

    tr_img = ImageSpecs(256).convert(img)
    tr_img = torch.unsqueeze(tr_img,0)
    # print(tr_img.size())


    out_var = model(tr_img)
assert model.xy_heatmaps[-1].size() == torch.Size([1, 17, 32, 32])
assert out_var.size() == torch.Size([1, 17, 3])

pred_skel_norm = ensure_homogeneous(out_var.squeeze(0).to(CPU, torch.float64), d=3)
print(pred_skel_norm)



# pred_skel_norm = ensure_homogeneous(out_var.squeeze(0).to(CPU, torch.float64), d=3)
#     pred_skel_denorm = dataset.denormalise_with_skeleton_height(
#         pred_skel_norm, example['camera'], example['transform_opts'])
#     pred_skel_image_space = example['camera'].project_cartesian(pred_skel_denorm)
#     pred_skel_camera_space = dataset.untransform_skeleton(pred_skel_denorm, example['transform_opts'])


