#!/usr/bin/env python3

"""Perform 3D pose inference on a single image.
The image is assumed to be centred on a human subject and scaled appropriately.
Since the camera intrinsics are not known, the skeleton will be shown in normalized form.
This means that bones may be warped due to non-reversed transformations.
"""

import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
import PIL.Image
import matplotlib.pylab as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from pose3d_utils.coords import ensure_cartesian

from margipose.cli import Subcommand
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import ImageSpecs
from margipose.models import load_model
from margipose.utils import seed_all, init_algorithms, plot_skeleton_on_axes3d

CPU = torch.device('cpu')


def parse_args():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='path to model file')
    parser.add_argument('--image', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='image file to infer pose from')

    parser.add_argument('--dir', type=str, metavar='DIR', default=argparse.SUPPRESS,
                        required=False,
                        help='directory of files to infer pose from')


    return parser.parse_args()


def main():


    args = parse_args()

    seed_all(12345)
    init_algorithms(deterministic=True)
    torch.set_grad_enabled(False)


    model = load_model(args.model).to(CPU).eval()

    #Obtains input spec from model and resizes the image
    input_specs: ImageSpecs = model.data_specs.input_specs
    image: PIL.Image.Image = PIL.Image.open(args.image, 'r')
    image.thumbnail((input_specs.width, input_specs.height))
    input_image = input_specs.convert(image).to(CPU, torch.float32)

    # print(input_image.size())
    # print(input_image[None, ...].size())
    # Make inference
    output = model(input_image[None, ...])[0]
    
    # Create location of normalized skeleton
    norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)
    # print(norm_skel3d)
    
    # plot
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2: Axes3D = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.imshow(input_specs.unconvert(input_image.to(CPU)))
    plot_skeleton_on_axes3d(norm_skel3d, CanonicalSkeletonDesc, ax2, invert=True)

    plt.show()



def draw_skeleton_2d(img, skel2d, skel_desc, mask=None, width=1):
    assert skel2d.size(-1) == 2, 'coordinates must be 2D'
    draw = ImageDraw.Draw(img)
    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id in range(skel_desc.n_joints):
        meta = get_joint_metadata(joint_id)
        color = (255, 0, 255)
        if meta['group'] == 'left':
            color = (0, 0, 255)
        if meta['group'] == 'right':
            color = (255, 0, 0)
        if mask is not None:
            if mask[joint_id] == 0 or mask[meta['parent']] == 0:
                color = (128, 128, 128)
        draw.line(
            [*skel2d[joint_id], *skel2d[meta['parent']]],
            color, width=width
        )


if __name__ == '__main__':
    main()