import numpy as np
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("../Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
#cap = cv2.VideoCapture(0)

def ComputeHistFlow(_flow):
    bin = 40

    hist = cv2.calcHist([_flow], [1, 0], None, [64, 64], [-bin, bin]*2)
    hist[hist[:,:]>np.std(hist)/2] = np.std(hist)/2

    # Normalize histogram
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


    # Display histogram
    plt.figure(num=2)
    plt.clf()
    plt.title("Hisogramme 2D de la probabilité jointe de Vx et Vy du flot optique")
    plt.xlabel("Vx")
    plt.ylabel("Vy")
    plt.imshow(hist, interpolation = 'nearest')
    plt.colorbar()
    plt.draw();

    return hist

def plot_correlation(corr, seuil):
    plt.figure(num=1, figsize=(4, 4))
    plt.clf()
    plt.rcParams["figure.figsize"] = (5,5)
    plt.plot(corr, 'b', linewidth = 0.5)
    plt.ylim([0, 1])
    plt.title("Correlation des histogrammes successives")
    plt.xlabel("Image")
    plt.ylabel("Correlation")
    plt.axhline(y=seuil, color='r', linestyle='-')
    plt.draw()
    plt.pause(0.0001)



ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris


index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris

flow = cv2.calcOpticalFlowFarneback(prvs,next,None,
                                    pyr_scale = 0.5,# Taux de réduction pyramidal
                                    levels = 3, # Nombre de niveaux de la pyramide
                                    winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                    iterations = 3, # Nb d'itérations par niveau
                                    poly_n = 7, # Taille voisinage pour approximation polynomiale
                                    poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées
                                    flags = 0)

Hist = np.zeros_like(ComputeHistFlow(flow))
Hist_p = np.zeros_like(Hist)

corr = []
seuil = 0.55

while(ret):
    index += 1
    Hist = Hist_p
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None,
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées
                                        flags = 0)

    Hist_p = ComputeHistFlow(flow)
    corr.append(cv2.compareHist(Hist, Hist_p, cv2.HISTCMP_CORREL)) #metrique = correlation
    plot_correlation(corr, seuil)

    cv2.namedWindow('Video frames')
    cv2.moveWindow('Video frames', 500, 0)
    cv2.imshow('Video frames',frame2)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('Hist%04d.png'%index,Hist_p)

    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)


cap.release()
cv2.destroyAllWindows()
