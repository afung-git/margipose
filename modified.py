# This is a custom scripted modified to output the inference of an individual image

import torch
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
    img = Image.open('./000002.jpg')
    # img = img.load()

    tr_img = ImageSpecs(256).convert(img)
    tr_img = torch.unsqueeze(tr_img,0)
    # print(tr_img.size())


    out_var = model(tr_img)
# assert model.xy_heatmaps[-1].size() == torch.Size([1, 17, 32, 32])
# assert out_var.size() == torch.Size([1, 17, 3])

pred_skel_norm = ensure_homogeneous(out_var.to(CPU, torch.float64), d=3)
print(pred_skel_norm)

# print(pred_skel_norm.numpy())
# coords = pred_skel_norm.numpy()
coords = np.squeeze(pred_skel_norm.numpy(), axis=0)
coords = np.rint((1+coords)*(255-0)/2)[:,:3]
coords_2d = coords[:,:2].astype(int)
# print(coords_2d)

blank = np.zeros((256,256))

for y,x in coords_2d:
    # print(x,y)
    blank[x,y] = 255.0
# print(blank)
# convert array to Image
# img = Image.fromarray(blank)
# img.show()