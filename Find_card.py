import numpy as np
import cv2 as cv

#MIN_AREA_CARD = 0.2*0.2
MIN_AREA_CARD = 0.1*0.1

#Below code is The modifed code by adding Minimum Area to calculate contours to run by taking tha Arguments from User
def find_cards(img):
  # Do adaptive threshold to find edges
  # Apply a gaussian blur just before to 
  # make sure it doesn't pick up noisy edges
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  gray = cv.GaussianBlur(gray, (21, 21), 0)
  th = cv.adaptiveThreshold(gray, 255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY, 11, 2)

  # Get negative to find contours
  th = 255 - th 

  # We can notice there is a large area around
  # the uno card. We will use to our advantage.
  # In th, currently the image is a the edges
  # detected in the image. We apply some dilatation
  # to close any contour AND when cropping, make
  # sure that we don't get anything outside the card

  # When finding the contours, we get two contours
  # around the card. We pick the most inner one.
  # The dilatation process will make sure the inner
  # contour is exclusively inside the card

  kernel = np.ones((3, 3), np.uint8)
  th = cv.morphologyEx(th, cv.MORPH_DILATE, kernel, iterations=8)

  # Find contour in thresholded image
  contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  # Card contours must have
  # 1. Have an certain minimal area
  # 2. Be the direct child of an external contour
  card_contours = []

  for i, cnt in enumerate(contours):
    # Check that contour is external
    if hierarchy[0][i][3] == -1:
      # # Check that contour has child
      if hierarchy[0][i][2] != -1:
        # Check that contour has minimal area
        if cv.contourArea(contours[hierarchy[0][i][2]]) > np.size(th) * MIN_AREA_CARD:
          card_contours.append(hierarchy[0][i][2])


  card_imgs = []
  card_outlines = []

  for idx in card_contours:
    cnt = contours[idx]
    # Find oriented bounding box around card
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)

    # Have the most top-left first, and also 
    # the contour should be clockwise from the 
    # previous functions
    idx_leftop = np.argmin([p[0]+p[1] for p in box])
    box_ordered = []
    for j in range(4):
      box_ordered.append(box[(idx_leftop+j)%4])
    box = np.array(box_ordered)

    # Estimate card width and height

    box_l11 = np.linalg.norm(box[0]-box[1])
    box_l12 = np.linalg.norm(box[2]-box[3])

    box_l21 = np.linalg.norm(box[1]-box[2])
    box_l22 = np.linalg.norm(box[3]-box[0])

    box_l1 = (box_l11+box_l12)/2
    box_l2 = (box_l21+box_l22)/2

    # Card is straight
    new_width, new_height = None, None
    if box_l1 < box_l2:
      new_points = np.array([
        [0, 0], [box_l1, 0], [box_l1, box_l2], [0, box_l2]])
      new_width = box_l1
      new_height = box_l2

    # Card is on its side
    else:
      new_points = np.array([[box_l2, 0], [box_l2, box_l1], [0, box_l1], [0, 0]])
      new_width = box_l2
      new_height = box_l1

    # Compute perspective transform matrix and get wrapped
    # image which is only the card
    M = cv.getPerspectiveTransform(np.float32(box), np.float32(new_points))
    roi = cv.warpPerspective(img, M, (round(new_width), round(new_height)))

    card_imgs.append(roi)
    card_outlines.append(box)

  return card_imgs, card_outlines



# Below code it to run Manually by the main function 
#def find_cards(images):
#  
#  
#  # Do adaptive threshold to find edges
#  # Apply a gaussian blur just before to 
#  # make sure it doesn't pick up noisy edges
#  gray = cv.cvtColor(images, cv.COLOR_BGR2GRAY)
#  gray = cv.GaussianBlur(gray, (21, 21), 0)
#  th = cv.adaptiveThreshold(gray, 255,
#      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#      cv.THRESH_BINARY, 11, 2)
#
#  # Get negative to find contours
#  th = 255 - th 
#
#  
#
#  # After getting two contours
#  # Pick the most inner one.
#  # Do dilatation process
#
#  kernel = np.ones((3, 3), np.uint8)
#  th = cv.morphologyEx(th, cv.MORPH_DILATE, kernel, iterations=10)
#
#  # Find biggest external contour which is 
#  # quadrilatel like
#  contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#  idx_biggest = np.argmax([cv.contourArea(c) for c in contours])
#  idx_inner = hierarchy[0][idx_biggest][2]
#  cnt = contours[idx_inner]
#
#  # cv.drawContours(images[i], contours, -1, (0, 0, 255), 2)
#  # cv.drawContours(images[i], contours, idx_inner, (0, 0, 255), 2)
#
#  # Find oriented bounding box around card
#  rect = cv.minAreaRect(cnt)
#  box = cv.boxPoints(rect)
#
#  # Have the most top-left first, and also 
#  # the contour should be clockwise from the 
#  # previous functions
#  idx_leftop = np.argmin([p[0]+p[1] for p in box])
#  box_ordered = []
#  card_imgs = []
#  card_outlines = []
#  for j in range(4):
#    box_ordered.append(box[(idx_leftop+j)%4])
#  box = np.array(box_ordered)
#
#
#  # cv.drawContours(images[i], [box], 0, (0, 0, 255), 2)
#  # plt.imshow(images[i])
#  # plt.show()
#
#  # Estimate card width and height
#
#  #   [0]   l11   [1]
#  #     ┌────────┐
#  #     │        │
#  # l22 │        │ l21
#  #     │        │
#  #     │        │
#  #     └────────┘
#  #   [3]   l12   [2]
#
#  box_l11 = np.linalg.norm(box[0]-box[1])
#  box_l12 = np.linalg.norm(box[2]-box[3])
#
#  box_l21 = np.linalg.norm(box[1]-box[2])
#  box_l22 = np.linalg.norm(box[3]-box[0])
#
#  box_l1 = (box_l11+box_l12)/2
#  box_l2 = (box_l21+box_l22)/2
#
#  # Card is straight
#  new_width, new_height = None, None
#  if box_l1 < box_l2:
#    new_points = np.array([
#      [0, 0], [box_l1, 0], [box_l1, box_l2], [0, box_l2]])
#    new_width = box_l1
#    new_height = box_l2
#
#  # Card is on its side
#  else:
#    new_points = np.array([[box_l2, 0], [box_l2, box_l1], [0, box_l1], [0, 0]])
#    new_width = box_l2
#    new_height = box_l1
#
#  # Compute perspective transform matrix and get wrapped
#  # image which is only the card
#  M = cv.getPerspectiveTransform(np.float32(box), np.float32(new_points))
#  roi = cv.warpPerspective(images, M, (round(new_width), round(new_height)))
#
#  card_imgs.append(roi)
#  card_outlines.append(box)
#  
#
#  return card_imgs, card_outlines
#