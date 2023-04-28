import cv2 as cv
import numpy as np


# Given an image finds the UNO color
# 1. The image is first resized with NEAREST
# 2. Do a K-means (K=3), should be white and UNO color
# 3. Pick color which is farthest from white


def find_color(img):

  # Resize image to have a width of 10
  # Here it's important that we resize using
  # "nearest". This assure that the color
  # are not mixed together during downsizing.
  fxy = 10/img.shape[1]
  small = cv.resize(img, None, fx=fxy, fy=fxy, interpolation=cv.INTER_NEAREST)

  # Apply k-means with k = 2 
  # The expected outcome is to have white
  # and the card color
  Z = small.reshape((-1,3))
  Z = np.float32(Z)

  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 3
  ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
  # Pick the color which is the farthest from pure white
  # this should be the card color
  idx_dominant = np.argmax([np.linalg.norm(c - np.array([255, 255, 255])) for c in center])
  dominant_color = center[idx_dominant]
  dominant_color = np.uint8(dominant_color)
  
  # Convert to HSV and extract hue
  dominant_color = cv.cvtColor(dominant_color[np.newaxis, np.newaxis, :], cv.COLOR_BGR2HSV)
  dominant_color = dominant_color[:,:,0]
  return dominant_color
