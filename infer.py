#!/usr/bin/env python3

"""Perform 3D pose inference on a single image or directory.
The image is assumed to be centred on a human subject and scaled appropriately.
3D skeleton output is only a normalized skeleton
"""

import argparse
import sys
import os
import PIL.Image
import matplotlib.pylab as plt
import numpy as np
import json
import torch
from mpl_toolkits.mplot3d import Axes3D
from pose3d_utils.coords import ensure_cartesian

from margipose.cli import Subcommand
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import ImageSpecs
from margipose.models import load_model
from margipose.utils import seed_all, init_algorithms, plot_skeleton_on_axes3d, plot_skeleton_on_axes

CPU = torch.device('cpu')
init_algorithms(deterministic=True)
torch.set_grad_enabled(False)
seed_all(12345)

def parse_args():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='path to model file')
    parser.add_argument('--mode', type=str, required=True, choices = ['I', 'D', 'd', 'i'],
                        help='infer single image or directory')

    parser.add_argument('--path', type=str, metavar='DIR', default=argparse.SUPPRESS,
                        required=True,
                        help='directory of files to infer pose from')


    return parser.parse_args()


def infer_single(model, img):
    # print(os.path.basename(img))
    filename = os.path.basename(img)
    filename_noext = os.path.splitext(filename)[0]
    # print(filename_noext)


    #Obtains input spec from model and resizes the image
    input_specs: ImageSpecs = model.data_specs.input_specs
    image: PIL.Image.Image = PIL.Image.open(img, 'r')
    image.thumbnail((input_specs.width, input_specs.height))
    input_image = input_specs.convert(image).to(CPU, torch.float32)


    # Make inference
    output = model(input_image[None, ...])[0]

    # Create location of normalized skeleton
    norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)
    coords = norm_skel3d.numpy()
    coords = np.rint((1+coords)*(255-0)/2)[:,:3]
    coords_2d = coords[:,:2].astype(int)
    img = input_specs.unconvert(input_image.to(CPU))
    img_2d = draw_joints_on_image(img, coords_2d)
    # print(norm_skel3d)
    
    # create visualization
    fig = plt.figure(1)
    plt_3d: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    plot_skeleton_on_axes3d(norm_skel3d, CanonicalSkeletonDesc, plt_3d, invert=True)
    # plt.show()

    # superimpose joints on input image
    
    #saving all outputs as image files with corresponding filename
    extent = plt_3d.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('./outputs/3d/' + filename, bbox_inches=extent) 
    img_2d.save('./outputs/' + filename)

    # create JSON file with all the joints saved
    joint_dic = {
        "head_top" : (coords_2d[0][0], coords_2d[0][1]),
        "neck" : (coords_2d[1][0], coords_2d[1][1]),
        "r_shoulder" : (coords_2d[2][0], coords_2d[2][1]),
        "r_elbow" : (coords_2d[3][0], coords_2d[3][1]),
        "r_wrist" : (coords_2d[4][0], coords_2d[4][1]),
        "l_shoulder" : (coords_2d[5][0], coords_2d[5][1]),
        "l_elbow" : (coords_2d[6][0], coords_2d[6][1]),
        "l_wrist" : (coords_2d[7][0], coords_2d[7][1]),
        "r_hip" : (coords_2d[8][0], coords_2d[8][1]),
        "r_knee" : (coords_2d[9][0], coords_2d[9][1]),
        "r_ankle" : (coords_2d[10][0], coords_2d[10][1]),
        "l_hip" : (coords_2d[11][0], coords_2d[11][1]),
        "l_knee" : (coords_2d[12][0], coords_2d[12][1]),
        "l_ankle" : (coords_2d[13][0], coords_2d[13][1]),
        "pelvis" : (coords_2d[14][0], coords_2d[14][1]),
        "spine" : (coords_2d[15][0], coords_2d[15][1]),
        "head" : (coords_2d[16][0], coords_2d[16][1]),
    }

    return joint_dic


def infer_dir():
    return 1

def main():

    args = parse_args()
    if (args.mode == 'I' or args.mode == 'i'):
        model = load_model(args.model).to(CPU).eval()
        joints_loc = infer_single(model, args.path)
    with open('data.json', 'w') as fp:
        json.dump(joints_loc, fp)
    # if(args.mode =='D' or args.mode == 'd'):
    #     infer_dir(args.path)


    

def draw_joints_on_image(img, coords_2d):
    for x,y in coords_2d:
        r = 3
        draw = PIL.ImageDraw.Draw(img)
        draw.ellipse((x-r, y-r, x+r, y+r), fill='yellow', outline='orange')
    return img

def output_to_JSON():
    return 1

if __name__ == '__main__':
    main()