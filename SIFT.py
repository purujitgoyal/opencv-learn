import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

# keypoints detector
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,(255,0,0),4)

plt.imshow(img, cmap='gray')
plt.show()

# keypoints descriptor
kp,des = sift.compute(gray,kp)

print "No of Key Points"
print len(kp)
print "Array of Descriptors"
print des.shape

