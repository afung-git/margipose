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
        # print(frameTotal, "Frames", type(frameTotal))
        print(frameHeight, 'x', frameWidth, 'Image size')
        # print(frameFourCC,  "CC", type(frameFourCC))
        count = 0
        frameArray = np.zeros((frameHeight, frameWidth, 3 ,frameTotal), dtype=np.uint8)
        # frameArray = np
        
        print(frameArray.shape[3], "Initial Frame Total")
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret == False:
                print(time.time()-start, "(s) to complete video to frame")
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
        
        # print(frameArray[:,:,:,:count].shape, "Actual Shape")
        print(fps, "FPS")
        # number of frames using CAP_PROP_FRAME_COUNT is only an estimate, count keeps track of actual frames
        print(count, 'Actaul Total Frames')
        # cv2.imwrite('./outputs/testimg01.jpg', frameArray[:,:,:,-1])
        # cv2.imwrite('./outputs/testimg01frame.jpg', frame)
        return(frameArray[:,:,:,:count], fps)

    @staticmethod
    def FrametoVid(frameArray, skel3DArray, fps, filename):
        # OPENCV can only have one VideoWriter object at a time

        outVid = cv2.VideoWriter('./outputs/'+filename+'.avi', 
            cv2.VideoWriter_fourcc(*'MJPG'), fps, (frameArray.shape[1],frameArray.shape[0]))

        for i in range(frameArray.shape[3]):
            outVid.write(frameArray[:,:,:,i][..., ::-1])
        outVid.release()
            
        outskel3D = cv2.VideoWriter('./outputs/'+'skel_'+filename+'.avi', 
            cv2.VideoWriter_fourcc(*'MJPG'), fps, (skel3DArray.shape[1],skel3DArray.shape[0]))

        for i in range(skel3DArray.shape[3]):
            outskel3D.write(skel3DArray[:,:,:,i])
        outskel3D.release()

    
# outtest = cv2.VideoWriter('./outputs/skel3d_test.avi', 
#             cv2.VideoWriter_fourcc(*'MJPG'), 23, (640,480))

# for i in range(30):
#     img = cv2.imread('./outputs/3d/skel3d_'+str(i)+'.jpg')
#     outtest.write(img)
# outtest.release()

