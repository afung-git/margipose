#!/usr/bin/env python3

"""Perform 3D pose inference on a single image or directory.
The image is assumed to be centred on a human subject and scaled appropriately.
3D skeleton output is only a normalized skeleton
"""

import argparse
import time
import sys
import os
import PIL
from PIL import ImageFont
import matplotlib.pylab as plt
import numpy as np
import json
import torch
from mpl_toolkits.mplot3d import Axes3D
from pose3d_utils.coords import ensure_cartesian
from MayaExporter import MayaExporter
from VideoFrames import VideoFrames

from margipose.cli import Subcommand
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import ImageSpecs
from margipose.models import load_model
from margipose.utils import seed_all, init_algorithms, plot_skeleton_on_axes3d, plot_skeleton_on_axes, angleBetween

CPU = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# CPU = torch.device('cpu')
init_algorithms(deterministic=True)
torch.set_grad_enabled(False)
torch.no_grad()
seed_all(12345)

def parse_args():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, metavar='FILE', default=argparse.SUPPRESS,
                        required=True,
                        help='path to model file')
    parser.add_argument('--mode', type=str, required=True, choices = ['I', 'D', 'd', 'i', 'v', 'V'],
                        help='infer single image or directory')

    parser.add_argument('--path', type=str, metavar='DIR', default=argparse.SUPPRESS,
                        required=True,
                        help='directory or files to infer pose from')
    return parser.parse_args()


def infer_joints(model, image):
    #Obtains input spec from model and resizes the image
    input_specs: ImageSpecs = model.data_specs.input_specs
    try:
        image: PIL.Image.Image = PIL.Image.open(image, 'r')
    except:
        pass
    if image.width != image.height:
        cropSize = min(image.width, image.height)
        image = image.crop((image.width/2 - cropSize/2, image.height/2 - cropSize/2,
                    image.width/2 + cropSize/2, image.height/2 + cropSize/2))

    if image.width < 256:
        image = image.resize((256, 256), PIL.Image.ANTIALIAS)

    image.thumbnail((input_specs.width, input_specs.height))
    input_image = input_specs.convert(image).to(CPU, torch.float32)
    # input_image = input_specs.convert(image).to(CPU, torch.cuda)

    # Make inference
    output = model(input_image[None, ...])[0]
    
    # Create location of normalized skeleton
    norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)
    # norm_skel3d = ensure_cartesian(output, d=3)
    coords = norm_skel3d.cpu().numpy()
    # if torch.cuda.is_available():    
        # coords = norm_skel3d.cpu().numpy()
    # else:
        # coords = norm_skel3d.numpy()
    coords_raw = coords
    coords_img = np.rint((1+coords)*(255-0)/2)[:,:3]
    coords_img = coords_img.astype(int)
    # print(coords_2d)
    img = input_specs.unconvert(input_image.to(CPU).cpu())
    # print(norm_skel3d)
       
    # create visualization of normalized skeleton
    fig = plt.figure(1)
    plt_3d: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    plot_skeleton_on_axes3d(norm_skel3d, CanonicalSkeletonDesc, plt_3d, invert=True)
    # plt.show()

    #saving all outputs as image files with corresponding filename
    fig.canvas.draw()
    fig_img = np.array(fig.canvas.renderer._renderer, np.uint8)[:,:,:3]
    # fig_img = fig_img[:,:,:3] 
    plt.close(fig)

    return (coords_img, coords_raw, img, fig_img)

# @jit # no significant improvements
def main():

    args = parse_args()
    if (args.mode == 'I' or args.mode == 'i'):
        filename = os.path.basename(args.path)
        filename_noext = os.path.splitext(filename)[0]
        model = load_model(args.model).to(CPU).eval()
        coords_img, coords_raw, img_input, img_skele3d = infer_joints(model, args.path)
        # print(img_skele3d.shape)

        img_skele3d = PIL.Image.fromarray(img_skele3d)
        
        # Calculate step angle
        vector1 = coords_img[8,:2] - coords_img[10,:2]
        vecto2 = (0,-1)

        
        print(angleBetween(vector1, vecto2)*180/np.pi, "right")

        vector1 = coords_img[11,:2] - coords_img[13,:2]
        # vector1=(.5,0,0)
        vecto2 = (0,-1)
        print(angleBetween(vector1, vecto2)*180/np.pi, "left")

        # draw_skele_on_image(img_input, coords_img[:,:2])
        # Used for qualitative evaluation
        draw_color_joints(img_input, coords_img[:,:2])


        joints_loc = output_to_JSON(coords_raw, filename_noext)
        # img_skele3d.show()
        
        img_skele3d.save('./outputs/3d/' + filename_noext + '.png')
        img_input.save('./outputs/' + filename)
        #with open('./outputs/joint_loc.json', 'w') as fp:
        #    json.dump(joints_loc, fp, indent=4)
        MayaExporter.WriteToMayaAscii('./outputs/3d/' + filename_noext + '.ma', joints_loc)



    if(args.mode =='D' or args.mode == 'd'):
        files = os.listdir(args.path)
        # print(files)
        start = time.time()
        model = load_model(args.model).to(CPU).eval()
        end = time.time()
        print(end-start, "To load Model")
        count = 0
        joints_loc_list = []
        for image in files:
            start = time.time()
            filename_noext = os.path.splitext(image)[0]
            coords_img, coords_raw, img_input, img_skele3d = infer_joints(model, args.path)
            # print(filename_noext)
            img_skele3d = PIL.Image.fromarray(img_skele3d)
                    
            # draw_skele_on_image(img_input, coords_img[:,:2])
            # Used for qualitative evaluation
            draw_color_joints(img_input, coords_img[:,:2])

            joints_loc = output_to_JSON(coords_raw, filename_noext)   

            img_skele3d.save('./outputs/3d/' + filename_noext + '.png')
            img_input.save('./outputs/' + image)
            joints_loc_list.append(joints_loc)
            count += 1
            print(time.time()-start, "(s)", "completed " + str(count) + "/" + str(len(files)))

        MayaExporter.WriteToMayaAscii('./outputs/3d/' + filename_noext + '.ma', joints_loc)

    if(args.mode =='V' or args.mode == 'v'):
        filename = os.path.basename(args.path)
        filename_noext = os.path.splitext(filename)[0]
        (frameArray, fps) = VideoFrames.ExtractFrames(args.path)

        frameArray = np.asarray(frameArray, dtype=np.uint8)
        skel3DArray = np.zeros((480, 640, 3, frameArray.shape[3]), dtype=np.uint8)
        finalFrameArray = np.zeros((256, 256, 3, frameArray.shape[3]), dtype=np.uint8)
        strideAngles = np.zeros((frameArray.shape[3],3))
        # print(frameArray.shape[3])

        start = time.time()
        model = load_model(args.model).to(CPU).eval()
        end = time.time()
        print(end-start, "(s) to load Model")
        sign = -1


        for i in range(frameArray.shape[3]):
        # for i in range(10):
            start = time.time()
 
            img = PIL.Image.fromarray(frameArray[:,:,:,i][..., ::-1])
            
            coords_img, coords_raw, img_scaled, skel3DArray[:,:,:,i] = infer_joints(model, img)


            # Calculate step angle
            vector1 = coords_img[8,:2] - coords_img[10,:2]
            vecto2 = (0,-1)

            rightStride = angleBetween(vector1, vecto2)*180/np.pi
            # print(angleBetween(vector1, vecto2)*180/np.pi, "right")

            vector1 = coords_img[11,:2] - coords_img[13,:2]
            # vector1=(.5,0,0)
            vecto2 = (0,-1)
            leftStride = angleBetween(vector1, vecto2)*180/np.pi
            # print(angleBetween(vector1, vecto2)*180/np.pi, "left")
            strideAngles[i,0] = (i+1)/fps
            strideAngles[i,1] = rightStride
            strideAngles[i,2] = leftStride

            draw_skele_on_image(img_scaled, coords_img[:,:2])
            # draw_evaluation_image(img_scaled, coords_img[:,:2], rightStride, leftStride)

            # img3d = PIL.Image.fromarray(skel3DArray[:,:,:,i])
            finalFrameArray[:,:,:,i] = np.array(img_scaled, dtype=np.uint8)
            # skel3DArray[:,:,:,i] = np.array(img3d, dtype=np.uint8)
            print(time.time()-start, "(s)", "frames completed " + str(i+1) + "/" + str(frameArray.shape[3]))
            # img3d.save('./outputs/3d/'+'skel3d_'+str(i)+'.jpg')
        # PIL.Image.SAVE(skel3DArray[:,:,:,-1])
        # plt.imsave()
        # plt.show()
        VideoFrames.FrametoVid(finalFrameArray, skel3DArray, fps, filename_noext)
        plt.subplot(2,1,1)
        plt.plot(strideAngles[:,0], strideAngles[:,1], color='red')
        plt.ylim(0,15)

        plt.subplot(2,1,2)
        plt.plot(strideAngles[:,0], strideAngles[:,2], color='blue')
        plt.ylim(0,15)

        plt.show()

        # print(strideAngles)



def draw_skele_on_image(img, coords):
    # print(coords.shape[0])
    r = 2
    linewidth = 2
    draw = PIL.ImageDraw.Draw(img)

    # draws center joints
    draw.line((coords[0,0], coords[0,1], coords[16,0], coords[16,1]), fill='white', width=linewidth)
    draw.line((coords[16,0], coords[16,1], coords[1,0], coords[1,1]), fill='white', width=linewidth)
    draw.line((coords[1,0], coords[1,1], coords[15,0], coords[15,1]), fill='white', width=linewidth)
    draw.line((coords[15,0], coords[15,1], coords[14,0], coords[14,1]), fill='white', width=linewidth)

    # draw right joints
    draw.line((coords[1,0], coords[1,1], coords[2,0], coords[2,1]), fill='red', width=linewidth)
    draw.line((coords[2,0], coords[2,1], coords[3,0], coords[3,1]), fill='red', width=linewidth)
    draw.line((coords[3,0], coords[3,1], coords[4,0], coords[4,1]), fill='red', width=linewidth)
    draw.line((coords[14,0], coords[14,1], coords[8,0], coords[8,1]), fill='red', width=linewidth)
    draw.line((coords[8,0], coords[8,1], coords[9,0], coords[9,1]), fill='red', width=linewidth)
    draw.line((coords[9,0], coords[9,1], coords[10,0], coords[10,1]), fill='red', width=linewidth)

    # draw left joints
    draw.line((coords[1,0], coords[1,1], coords[5,0], coords[5,1]), fill='blue', width=linewidth)
    draw.line((coords[5,0], coords[5,1], coords[6,0], coords[6,1]), fill='blue', width=linewidth)
    draw.line((coords[6,0], coords[6,1], coords[7,0], coords[7,1]), fill='blue', width=linewidth)
    draw.line((coords[14,0], coords[14,1], coords[11,0], coords[11,1]), fill='blue', width=linewidth)
    draw.line((coords[11,0], coords[11,1], coords[12,0], coords[12,1]), fill='blue', width=linewidth)
    draw.line((coords[12,0], coords[12,1], coords[13,0], coords[13,1]), fill='blue', width=linewidth)

    for i in range(coords.shape[0]):
        # r = 1
        (x,y) = coords[i,:]
        draw.ellipse((x-r, y-r, x+r, y+r), fill='white', outline='gray')

def draw_color_joints(img, coords):
    # draws color coded joints for quick qualatative evaluation
    draw = PIL.ImageDraw.Draw(img)
    center = 'white'
    right = (251, 18, 34)  
    left = (0, 0, 255)
    font = ImageFont.truetype('./fonts/Roboto-Bold.ttf', size=10)
# 0-4
        # 'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        # # 5-9
        # 'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
        # # 10-14
        # 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
        # # 15-16
        # 'spine', 'head'

# draws center joints
    draw.text((coords[0,0], coords[0,1]), "HT", fill=center, font=font)
    draw.text((coords[1,0], coords[1,1]), "N", fill=center, font=font)
    draw.text((coords[14,0], coords[14,1]), "P", fill=center, font=font)
    draw.text((coords[15,0], coords[15,1]), "S", fill=center, font=font)
    draw.text((coords[16,0], coords[16,1]), "H", fill=center, font=font)

    # draw right joints
    draw.text((coords[2,0], coords[2,1]), 'RS', fill=right, font=font)
    draw.text((coords[3,0], coords[3,1]), 'RE', fill=right, font=font)
    draw.text((coords[4,0], coords[4,1]), 'RE', fill=right, font=font)
    draw.text((coords[8,0], coords[8,1]), 'RH', fill=right, font=font)
    draw.text((coords[9,0], coords[9,1]), 'RK', fill=right, font=font)    
    draw.text((coords[10,0], coords[10,1]), 'RA', fill=right, font=font)

    # draw left joints
    draw.text((coords[5,0], coords[5,1]), 'LS', fill=left, font=font)
    draw.text((coords[6,0], coords[6,1]), 'LE', fill=left, font=font)
    draw.text((coords[7,0], coords[7,1]), 'LW', fill=left, font=font)
    draw.text((coords[11,0], coords[11,1]), 'LH', fill=left, font=font)
    draw.text((coords[12,0], coords[12,1]), 'LK', fill=left, font=font)
    draw.text((coords[13,0], coords[13,1]), 'LA', fill=left, font=font)


def draw_evaluation_image(img, coords, Rangles, Langle):
    # print(coords.shape[0])
    r = 2
    linewidth = 2
    draw = PIL.ImageDraw.Draw(img)
    right = (251, 18, 34)  
    left = (0, 0, 255)
    font = ImageFont.truetype('./fonts/Roboto-Bold.ttf', size=10)


    # draw right leg
    draw.ellipse((coords[8,0]-r, coords[8,1]-r, coords[8,0]+r, coords[8,1]+r), fill='white', outline='gray')
    draw.ellipse((coords[10,0]-r, coords[10,1]-r, coords[10,0]+r, coords[10,1]+r), fill='white', outline='gray')
    draw.line((coords[8,0], coords[8,1], coords[10,0], coords[10,1]), fill='red', width=linewidth)
    draw.text((coords[10,0], coords[10,1] + 70), '{0:.2f}'.format(Rangles), fill=right, font=font)

    # draw left joints
    draw.ellipse((coords[11,0]-r, coords[11,1]-r, coords[11,0]+r, coords[11,1]+r), fill='white', outline='gray')
    draw.ellipse((coords[13,0]-r, coords[13,1]-r, coords[13,0]+r, coords[11,1]+r), fill='white', outline='gray')
    draw.line((coords[11,0], coords[11,1], coords[13,0], coords[13,1]), fill='blue', width=linewidth)
    draw.text((coords[13,0], coords[13,1] - 100), '{0:.2f}'.format(Langle), fill=left, font=font)


   

def output_to_JSON(coords, filename):
    # create JSON file with all the joints saved

    coords = coords*100
    joints_loc = {
        str("root_" + filename): {
            "pelvis" : {
                "t": coords[14],
                        "r_hip" : {
                            "t": coords[8],
                            "r_knee" : {
                                "t": coords[9],
                                "r_ankle" : {
                                    "t":coords[10],
                                    },
                                },
                            },
                        "l_hip" : {
                            "t": coords[11],
                            "l_knee" : {
                                "t": coords[12],
                                "l_ankle" : {
                                    "t": coords[13],
                                    },
                                },
                            },
                "spine" : {
                    "t": coords[15],
                    "neck": {
                        "t": coords[1],
                        "head" : {
                            "t": coords[16],
                            "head_top":{
                                "t": coords[0],
                                },
                            },
                        "r_shoulder" : {
                            "t": coords[2],
                            "r_elbow" : {
                                "t": coords[3],
                                "r_wrist" : {
                                    "t": coords[4],
                                    },
                                },
                            },
                        "l_shoulder" : {
                            "t": coords[5],
                            "l_elbow" : {
                                "t": coords[6],
                                "l_wrist" : {
                                    "t": coords[7],
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

    return joints_loc

if __name__ == '__main__':
    main()