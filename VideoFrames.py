import cv2
import os
import numpy as np

class VideoFrames:
    @staticmethod
    def ExtractFrames(videoLoc, saveDir):
        vid = cv2.VideoCapture(videoLoc)
        fps = vid.get(cv2.CAP_PROP_FPS)
        print(fps, "fps")
        # print(vid)
        count = 0
        # print(vid.isOpened())
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret == False:
                print("goes here")
                break
            # if count == 21: # for troubleshooting
            #     break
            cv2.imwrite(saveDir + str(count) + '.jpg', frame)
            count +=1
        vid.release()
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
        

