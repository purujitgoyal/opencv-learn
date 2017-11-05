import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import data, img_as_float, color
from skimage.util import random_noise

img = cv2.imread('lena.png',0)
original = img_as_float(img)
sigma = 0.155

noisy = random_noise(original, var=sigma**2)


f = np.fft.fft2(noisy)
fshift = np.fft.fftshift(f)
original_magnitude_spectrum = 20*np.log(np.abs(fshift))
rows, cols = noisy.shape
crow,ccol = rows//2 , cols//2

mask = np.zeros((rows,cols),np.uint8)
msize = 30
mask[crow-msize:crow+msize, ccol-msize:ccol+msize] = 1


fshift = fshift * mask
final_magnitude_spectrum = 20*np.log(np.abs(fshift))
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(231),plt.imshow(noisy, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(original_magnitude_spectrum, cmap = 'gray')
plt.title('Input Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(mask, cmap = 'gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(final_magnitude_spectrum, cmap = 'gray')
plt.title('Spectrum afer HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

plt.show()
