import cv2
import os
import numpy as np
import PIL
import torch
import time




class VideoFrames:
    @staticmethod
    def ExtractFrames(videoLoc):
        start = time.time()
        vid = cv2.VideoCapture(videoLoc)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        frameTotal = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frameHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frameFourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        print(fps, "fps", type(fps))
        print(frameTotal, "frames", type(frameTotal))
        print(frameHeight, frameWidth, type(frameHeight))
        # print(frameFourCC,  "CC", type(frameFourCC))
        count = 0
        frameArray = np.zeros((frameHeight, frameWidth, 3 ,frameTotal), dtype=np.uint8)
        # frameArray = np
        
        print(frameArray.shape, "initial shape")
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret == False:
                print("goes here")
                break
            # if count == 5: # for troubleshooting
            #     break
            if count > frameTotal:
                frame = np.expand_dims(frame, axis=-1)
                frameArray = np.append(frameArray, frame, axis=-1)
                continue
            frameArray[:,:,:,count] = frame
            count +=1
        vid.release()
        
        print(frameArray.shape, "final shape")
        print(fps, "fps")
        end = time.time()
        print(end-start, "(s) to complete vid to frame")
        # cv2.imwrite('./outputs/testimg01.jpg', frameArray[:,:,:,-1])
        # cv2.imwrite('./outputs/testimg01frame.jpg', frame)
        return(frameArray, fps)

    @staticmethod
    def FrametoVid(frameArray, skel3DArray, fps, filename):
        frameWidth = frameHeight = 256
        skel3DWidth = 480
        skel3DHeight = 640
        outVid = cv2.VideoWriter('./outputs/'+filename+'.avi', 
            cv2.VideoWriter_fourcc(*'DIVX'), fps, (frameWidth,frameHeight))

        outskel3D = cv2.VideoWriter('./outputs/3d/skel3D_'+filename+'.avi', 
            cv2.VideoWriter_fourcc(*'DIVX'), fps, (skel3DWidth,skel3DHeight))

        # frames = len(os.listdir(frameLoc))
        for i in range(frameArray.shape[3]):
            outVid.write(frameArray[:,:,:,i][..., ::-1])
            outskel3D.write(skel3DArray[:,:,:,i][..., ::-1])
        outVid.release()
        outskel3D.release()

    

    

