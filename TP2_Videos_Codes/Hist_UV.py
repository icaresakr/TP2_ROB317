import numpy as np
import cv2
import matplotlib.pyplot as plt


capt = cv2.VideoCapture("../Videos/Extrait3-Vertigo-Dream_Scene(320p).m4v")


def ComputeHist(_capt):
	bins = 256
	ret, frame = _capt.read()
	if ret:

		numPixels = np.prod(frame.shape[:2])
		
		#Convert image to YV space
		yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)


		hist = cv2.calcHist([yuv_image], [1, 2], None, [bins]*2, [0, 256]*2)
		
		# Normalize histogram
		hist[:,:] = np.log(hist[:,:])
		hist_norm = np.clip(hist, 0, np.max(hist))
		hist_norm[:,:] = (hist_norm[:,:]/np.max(hist_norm))
		#cv2.normalize(hist, hist_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


		# Display the video frames
		cv2.namedWindow('Video frames')     
		cv2.moveWindow('Video frames', 500,0)  
		cv2.imshow('Video frames', frame)
		
		
		# Display histogram
		plt.figure(num=2)
		plt.clf()
		plt.title("Hisogramme 2D de la probabilité jointe de u et v")
		plt.xlabel("V")
		plt.ylabel("U")
		plt.imshow(hist_norm,interpolation = 'nearest')
		plt.colorbar()
		plt.draw();
	
		return hist_norm
		
def plot_correlation(corr,seuil):
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


hist = np.zeros_like(ComputeHist(capt))
hist_p = np.zeros_like(ComputeHist(capt))
corr = []
seuil = 0.35
index = 0
ret = True
while ret:
	ret, frame = capt.read()
	if ret:
		
		hist = hist_p
		hist_p = ComputeHist(capt)
		
		corr.append(cv2.compareHist(hist, hist_p, cv2.HISTCMP_CORREL))
		
		plot_correlation(corr, seuil)
		
		index+=1
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('Frame%04d.png'%index,frame)
		cv2.imwrite('Hist%04d.png'%index,hist_p)



capt.release()
cv2.destroyAllWindows()
