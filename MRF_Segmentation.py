from PIL import Image
import numpy
from pylab import *
from scipy.cluster.vq import *
from scipy.signal import *
import cv2
import scipy
from matplotlib import pyplot as plt

def main():
	img = Image.open('home.jpg')
	img = numpy.array(img)
	
	# If grayscale add one dimension for easy processing
	if (img.ndim == 2):
		img = img[:,:,axis]

	nlevels = 4	
	lev = initkmean(img, nlevels)

	# MRF ICM
	win_dim = 256
	while (win_dim>7):
		print win_dim
		loc_avg = local_average(img, lev, nlevels, win_dim)
		lev = MRF(img, lev, loc_avg, nlevels)
		win_dim = win_dim/2

	# scipy.misc.imsave('lev.png',lev*20)
	plt.imshow(lev*20, cmap='gray')
	plt.show()

def initkmean(img, nlevels):
	obs = reshape(img,(img.shape[0]*img.shape[1], -1))	
	obs = whiten(obs)

	(centroids, lev) = kmeans2(obs, nlevels)
	lev = lev.reshape(img.shape[0], img.shape[1])
	return lev

def delta(a,b):
	if (a == b):
		return -1
	else:
		return 1	

def MRF(obs, lev, loc_avg, nlevels):
	(M,N) = obs.shape[0:2]
	for i in range(M):
		for j in range(N):
			cost=[energy(k,i,j, obs, lev, loc_avg) for k in range(nlevels)]
			lev[i,j] = cost.index(min(cost))
	return lev
			
def energy(pix_lev,i, j, obs,lev,loc_avg):
	beta = 0.5
	std = 7
	cl = clique(pix_lev,i,j,lev)
	closeness = numpy.linalg.norm(loc_avg[i,j,:,pix_lev]-obs[i,j,:])
	return beta*cl+closeness/std**2

def local_average(obs, lev, nlevels, win_dim):
	mask = numpy.ones((win_dim,win_dim))/win_dim**2
	loc_avg = ones((obs.shape+(nlevels,)))

	for i in range(obs.shape[2]):
		for j in range(nlevels):
			temp = (obs[:,:,i]*(lev == j))
			loc_avg[:,:,i,j] = fftconvolve(temp, mask, mode='same')
	return loc_avg

def clique(pix_lev, i, j, lev):
	(M,N)=lev.shape[0:2]

	if (i == 0 and j == 0):
		neighbor = [(0,1), (1,0)]
	elif (i == 0 and j == N-1):
		neighbor = [(0,N-2), (1,N-1)]
	elif (i == M-1 and j == 0):
		neighbor = [(M-1,1), (M-2,0)]
	elif (i == M-1 and j == N-1):
		neighbor = [(M-1,N-2), (M-2,N-1)]
	elif (i == 0):
		neighbor = [(0,j-1), (0,j+1), (1,j)]
	elif (i == M-1):
		neighbor = [(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif (j == 0):
		neighbor = [(i-1,0), (i+1,0), (i,1)]
	elif (j == N-1):
		neighbor = [(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:
		neighbor = [(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
	
	return sum(delta(pix_lev,lev[i]) for i in neighbor)

if (__name__ == "__main__"):
	main()	