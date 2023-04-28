import cv2 as cv
import numpy as np


def find_number_contour(img):
  # Do adaptive threshold to find the contours
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  gray = cv.GaussianBlur(gray, (1, 7), 0)
  th = cv.adaptiveThreshold(gray, 255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY, 11, 11)

  # Get negative to find contours
  th = 255 - th 

  # Some colors might not get clearly defined
  # during adaptive threshold. To fill any holes,
  # we do a morphological dilation
  kernel = np.ones((3, 5), np.uint8)
  th = cv.morphologyEx(th, cv.MORPH_DILATE, kernel)

  # Only find the external contour, the number in the
  # center should be an external contour
  contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  #Find contour closest to center
  center = np.array([th.shape[1]/2, th.shape[0]/2])
  contour_centers = []
  for cnt in contours:
    M = cv.moments(cnt)
    if M['m00'] != 0 and cv.contourArea(cnt) > 20:
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      contour_centers.append(np.array([cx, cy]))
    else:
      contour_centers.append(np.array([0, 0]))

  if len(contour_centers) == 0:
    return None

  closest_idx = np.argmin([np.linalg.norm(c - center) for c in contour_centers])
  cnt = contours[closest_idx]

  # cnt is the number coutour


  return cnt
