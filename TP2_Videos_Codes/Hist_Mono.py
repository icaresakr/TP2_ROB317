import numpy as np
import cv2
import matplotlib.pyplot as plt


capt = cv2.VideoCapture("../Videos/Extrait3-Vertigo-Dream_Scene(320p).m4v")

def ComputeHistMono(_capt):
    bin = 64
    ret, frame = _capt.read()
    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bin], [0,256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        #display the video frames
        cv2.namedWindow('Video frames')
        cv2.moveWindow('Video frames', 500, 0) # move it so we can see it
        cv2.imshow('Video frames', gray)

        #display the 1D histogram
        plt.figure(num=2)
        plt.clf()
        plt.title("Hisogramme 2D de la probabilité jointe de u et v")
        plt.xlabel("Grayscale Value")
        plt.ylabel("nb of Pixels")
        plt.plot(hist)
        plt.draw()

        return hist


def plot_correlation(err, seuil):
    plt.figure(num=1, figsize=(4, 4))
    plt.clf()
    plt.rcParams["figure.figsize"] = (5,5)
    plt.plot(err, 'b', linewidth = 0.5)
    plt.axhline(y=seuil, color='r', linestyle='-')
    plt.ylim([0, 1])
    plt.title("Correlation des histogrammes successives")
    plt.xlabel("Image")
    plt.ylabel("Correlation")
    plt.draw()
    plt.pause(0.0001)


hist = np.zeros_like(ComputeHistMono(capt))
hist_p = np.zeros_like(ComputeHistMono(capt))
corr = []

index = 0
ret = True
while ret:
    ret, frame = capt.read()
    if ret:

        hist = hist_p
        hist_p = ComputeHistMono(capt)

        corr.append(cv2.compareHist(hist, hist_p, cv2.HISTCMP_CORREL))

        plot_correlation(corr, 0.8)

        index+=1

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame%04d.png'%index,frame)
        cv2.imwrite('Hist%04d.png'%index,hist_p)



capt.release()
cv2.destroyAllWindows()