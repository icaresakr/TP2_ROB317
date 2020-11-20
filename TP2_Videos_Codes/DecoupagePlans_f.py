
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from numpy import shape
from skimage import data, exposure

cap = cv2.VideoCapture("../Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

RealCuts = [250,479,511,600,653,691, 1114, 1181, 1310, 1415, 1517, 1565, 1712, 1781,1864, 1989, 2047,2166,2216,2278,2442,2512,2559,2637,2714,2765,2838,3020,3094,3131,3162] #as seen by eyes


def ComputeHist(frame):
    bins = 256

    numPixels = np.prod(frame.shape[:2])

    #Convert image to YV space
    yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)


    hist = cv2.calcHist([yuv_image], [1, 2], None, [bins]*2, [0, bins]*2)

    # Normalize histogram
    #hist[:,:] = np.log(hist[:,:])
    hist_norm = np.clip(hist, 0, np.max(hist))
    hist_norm[:,:] = (hist_norm[:,:]/np.max(hist_norm))
    #cv2.normalize(hist, hist_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_norm



index = 0

ret, frame2 = cap.read()

hist = ComputeHist(frame2)
hist_p = ComputeHist(frame2)
corr = []

index = 0
cutIndexes = []

seuil = 0.35

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        index+=1

        hist = hist_p
        hist_p = ComputeHist(frame)

        val_corr = cv2.compareHist(hist, hist_p, cv2.HISTCMP_CORREL)

        corr.append(val_corr)

        if(val_corr<seuil):
            # on a donc trouvé un changement de plan
            cutIndexes.append(index)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame%04d.png'%index,frame)
        cv2.imwrite('Hist%04d.png'%index,hist_p)

print(len(cutIndexes))
cap.release()
cv2.destroyAllWindows()
