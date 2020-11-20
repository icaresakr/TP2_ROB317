import numpy as np
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("../Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

RealCuts = [250,479,511,600,653,691, 1114, 1181, 1310, 1415, 1517, 1565, 1712, 1781,1864, 1989, 2047,2166,2216,2278,2442,2512,2559,2637,2714,2765,2838,3020,3094,3131,3162] #as seen by eyes


def ComputeHist(frame):
    bins = 256

    numPixels = np.prod(frame.shape[:2])

    #Convert image to YV space
    yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)


    hist = cv2.calcHist([yuv_image], [1, 2], None, [bins]*2, [0, 256]*2)

    # Normalize histogram
    #hist[:,:] = np.log(hist[:,:])
    hist_norm = np.clip(hist, 0, np.max(hist))
    hist_norm[:,:] = (hist_norm[:,:]/np.max(hist_norm))
    #cv2.normalize(hist, hist_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_norm




corr = []

index = 0


n = len(RealCuts)
keyIndexes =[]

print(n)

for i in range(n):
    #print(i)
    #pour chaque plan
    ret, frame = cap.read()
    if(ret):
        hist_p = ComputeHist(frame)
        mean_hist = ComputeHist(frame)
        hists = [hist_p]

        if i==0:
            for j in range(0, RealCuts[i]-1):
                ret, frame = cap.read()
                #compute mean histogram

                hists.append(hist_p)
                hist_p = ComputeHist(frame)
                mean_hist = mean_hist+hist_p
            mean_hist = mean_hist/(RealCuts[i])
            keyIndex = 0
            max_corr = 0
            for k in range(len(hists)):
                corr = cv2.compareHist(hists[k], mean_hist, cv2.HISTCMP_CORREL)
                if(corr>max_corr):
                    max_corr = corr
                    keyIndex = k

            keyIndexes.append(keyIndex)

        else:
            for j in range(RealCuts[i-1], RealCuts[i]-1):
                #print(j)
                ret, frame = cap.read()
                #compute mean histogram
                hists.append(hist_p)
                hist_p = ComputeHist(frame)
                mean_hist = mean_hist+hist_p
            mean_hist = mean_hist/(RealCuts[i]-RealCuts[i-1])
            keyIndex = RealCuts[i-1]
            max_corr = 0
            for k in range(len(hists)):
                corr = cv2.compareHist(hists[k], mean_hist, cv2.HISTCMP_CORREL)
                if(corr>max_corr):
                    max_corr = corr
                    keyIndex = RealCuts[i-1]+k

            keyIndexes.append(keyIndex)

print(keyIndexes)

cap.release()

#replay video to save key_images:
cap = cv2.VideoCapture("../Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

index = 0

ret, frame2 = cap.read()

while(ret):

    if(index in keyIndexes):
        #we found the key image frame, save it
        cv2.imwrite('./key_images_color/Key_image%04d.png'%index,frame2)
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
