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


def infer_joints(model, img):
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
    coords = coords.astype(int)
    # print(coords_2d)
    img = input_specs.unconvert(input_image.to(CPU))
    # print(norm_skel3d)
    
    # create visualization of normalized skeleton
    fig = plt.figure(1)
    plt_3d: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    plot_skeleton_on_axes3d(norm_skel3d, CanonicalSkeletonDesc, plt_3d, invert=True)
    # plt.show()

    #saving all outputs as image files with corresponding filename
    fig.canvas.draw()
    fig_img = np.array(fig.canvas.renderer._renderer, np.uint8)

    return (coords, img, fig_img)


def main():

    args = parse_args()
    if (args.mode == 'I' or args.mode == 'i'):
        filename = os.path.basename(args.path)
        filename_noext = os.path.splitext(filename)[0]
        model = load_model(args.model).to(CPU).eval()
        coords, img_input, img_skele3d = infer_joints(model, args.path)
        # print(img_skele3d)

        img_skele3d = PIL.Image.fromarray(img_skele3d)
        image_joints = draw_joints_on_image(img_input, coords)
        joints_loc = output_to_JSON(coords[:,:2], filename_noext)
        # img_skele3d.show()
        
        img_skele3d.save('./outputs/3d/' + filename_noext + '.png')
        image_joints.save('./outputs/' + filename)
        with open('./outputs/joint_loc.json', 'w') as fp:
            json.dump(joints_loc, fp, indent=4)
    if(args.mode =='D' or args.mode == 'd'):
        files = os.listdir(args.path)
        print(files)



def draw_joints_on_image(img, coords):
    for x,y in coords[:,:2]:
        r = 3
        draw = PIL.ImageDraw.Draw(img)
        draw.ellipse((x-r, y-r, x+r, y+r), fill='yellow', outline='orange')
    return img

def output_to_JSON(coords_2d, filename):
    # create JSON file with all the joints saved
    joints_loc = {
        filename: {
            "head_top":{"x":int(coords_2d[0][0]), "y": int(coords_2d[0][1])},
            "neck": {"x":int(coords_2d[1][0]), "y": int(coords_2d[1][1])},
            "r_shoulder" : {"x":int(coords_2d[2][0]), "y": int(coords_2d[2][1])},
            "r_elbow" : {"x":int(coords_2d[3][0]), "y": int(coords_2d[3][1])},
            "r_wrist" : {"x":int(coords_2d[4][0]), "y": int(coords_2d[4][1])},
            "l_shoulder" : {"x":int(coords_2d[5][0]), "y": int(coords_2d[5][1])},
            "l_elbow" : {"x":int(coords_2d[6][0]), "y": int(coords_2d[6][1])},
            "l_wrist" : {"x":int(coords_2d[7][0]), "y": int(coords_2d[7][1])},
            "r_hip" : {"x":int(coords_2d[8][0]), "y": int(coords_2d[8][1])},
            "r_knee" : {"x":int(coords_2d[9][0]), "y": int(coords_2d[9][1])},
            "r_ankle" : {"x":int(coords_2d[10][0]), "y": int(coords_2d[10][1])},
            "l_hip" : {"x":int(coords_2d[11][0]), "y": int(coords_2d[11][1])},
            "l_knee" : {"x":int(coords_2d[12][0]), "y": int(coords_2d[12][1])},
            "l_ankle" : {"x":int(coords_2d[13][0]), "y": int(coords_2d[13][1])},
            "pelvis" : {"x":int(coords_2d[14][0]), "y": int(coords_2d[14][1])},
            "spine" : {"x":int(coords_2d[15][0]), "y": int(coords_2d[15][1])},
            "head" : {"x":int(coords_2d[16][0]), "y": int(coords_2d[16][1])},
        }
    }
    return joints_loc

if __name__ == '__main__':
    main()