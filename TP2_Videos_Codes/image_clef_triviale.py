
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from numpy import shape
from skimage import data, exposure


cap = cv2.VideoCapture("../Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

RealCuts = [250,479,511,600,653,691, 1114, 1181, 1310, 1415, 1517, 1565, 1712, 1781,1864, 1989, 2047,2166,2216,2278,2442,2512,2559,2637,2714,2765,2838,3020,3094,3131,3162] #as seen by eyes

nbFrames = 3168

def ComputeKeyIndexes(_cuts):
    indexes = []
    for j in range(len(_cuts)):
        if j==0:
            index = int(_cuts[j]/2)
        else:
            index = _cuts[j-1] + int((_cuts[j] - _cuts[j-1])/2)
        indexes.append(index)

    return(indexes)


keyIndexes = ComputeKeyIndexes(RealCuts)

index = 0

ret, frame2 = cap.read()

while(ret):

    if(index in keyIndexes):
        #we found the key image frame, save it
        cv2.imwrite('./key_images_trivial/Key_image%04d.png'%index,frame2)
        cv2.namedWindow('Video frames')
        cv2.moveWindow('Video frames', 500, 0)
        cv2.imshow('Video frames',frame2)


    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)

    ret, frame2 = cap.read()
    if (ret):
        index += 1


cap.release()
cv2.destroyAllWindows()