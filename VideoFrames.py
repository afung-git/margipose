import cv2
import os
import numpy as np
import PIL
import torch
from margipose.data_specs import ImageSpecs
from pose3d_utils.coords import ensure_cartesian
from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.utils import seed_all, init_algorithms, plot_skeleton_on_axes3d, plot_skeleton_on_axes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
from margipose.models import load_model

# import infer as infer



class VideoFrames:
    @staticmethod
    def infer_joints(self, model, img):
        
        #Obtains input spec from model and resizes the image
        input_specs: ImageSpecs = model.data_specs.input_specs
        image: PIL.Image.Image = PIL.Image.open(img, 'r')

        if image.width != image.height:
            cropSize = min(image.width, image.height)
            image = image.crop((image.width/2 - cropSize/2, image.height/2 - cropSize/2,
                        image.width/2 + cropSize/2, image.height/2 + cropSize/2))

        image.thumbnail((input_specs.width, input_specs.height))
        input_image = input_specs.convert(image).to(CPU, torch.float32)

        # Make inference
        output = model(input_image[None, ...])[0]

        # Create location of normalized skeleton
        norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)
        coords = norm_skel3d.numpy()
        coords_raw = coords
        coords_img = np.rint((1+coords)*(255-0)/2)[:,:3]
        coords_img = coords_img.astype(int)
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

        return (coords_img, coords_raw, img, fig_img)

    @staticmethod
    def ExtractFrames(videoLoc, saveDir):
        vid = cv2.VideoCapture(videoLoc)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frameHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameFourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        print(fps, "fps", type(fps))
        print(frameCount, "frames", type(frameCount))
        print(frameHeight, frameWidth, type(frameHeight))
        # print(frameFourCC,  "CC", type(frameFourCC))
        count = 0
        frameArray = np.zeros((frameHeight, frameWidth, 3 ,frameCount))
        print(frameArray.shape)
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret == False:
                print("goes here")
                break
            if count == 5: # for troubleshooting
                break
            # print(frame.shape)
            # print(type(frame))
            cv2.imwrite(saveDir + str(count) + '.jpg', frame)
            frameArray[:,:,:,count] = frame
            print(frame.shape)
            # plt.imshow(frameArray[:,:,1,count])
            # plt.imshow(frame[:,:,0])
            # plt.show()
            print(np.array_equal(frame, frameArray[:,:,:,count]))
            count +=1
        vid.release()
        print(frameArray[:,:,:,0].shape, "here <----")
        # img =PIL.Image.fromarray(frameArray[:,:,:,2],'RGB')
        # img.show()
        # plt.im
        # cv2.cv2.imshow("image", frameArray[:,:,:,0])
        return(fps)

    @staticmethod
    def FrametoVid(frameLoc, outputName, fps):
        frame_width = frame_height = 256
        out = cv2.VideoWriter('./outputs/'+outputName+'.avi', 
            cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width,frame_height))

        frames = len(os.listdir(frameLoc))
        for i in range(frames):
            # print(i)
            img = cv2.imread(frameLoc + str(i) +'.jpg')
            # image = cv2.cv2.ConvertTO
            out.write(img)
        out.release()

    @staticmethod
    def AllinOne(self, videoLoc, model):
        CPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('passed')
        outputName = 'test'
        model = load_model(model).to(CPU).eval()
        vid = cv2.VideoCapture(videoLoc)
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        frameCount =  0
        frame_width = frame_height = 256
        out = cv2.VideoWriter('./outputs/'+outputName+'.avi', 
            cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width,frame_height))
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret == False:
                print("goes here")
                break
            if frameCount == 5: # for troubleshooting
                break
            print(frame.shape)
            # infer_joints()
            img = PIL.Image.fromarray(frame)
            coords_img, coords_raw, img_input, img_skele3d = self.infer_joints(model, img)
        
            
            
            print(coords_img[:,:2])
        #     # print(type(frame))
        #     # cv2.imwrite(saveDir + str(count) + '.jpg', frame)
        #     infer.infer_joints()
            frameCount +=1
        vid.release()
        out.release()

    

