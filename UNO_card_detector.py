import cv2 as cv
import numpy as np
import glob
import os
import argparse

#calling functions from files within the directory
from Find_card import find_cards 
from Find_contour import find_number_contour
from Find_Color import find_color

# This is the minimum area
MIN_AREA_CARD = 0.1*0.1

# Card size after normlization
CARD_WIDTH = 400
CARD_RATIO = 1.5
CARD_HEIGHT = 600


# the below arguments were added to allow the user to have the option
# between the webcam or static images while running the code

parser = argparse.ArgumentParser(description='UNO card recognizer')
parser.add_argument('--images', help="Input CARDS")
parser.add_argument('--camdevice', type=int, default=0, help="Camera number")
parser.add_argument('--mode', choices=['livecam', 'staticimg'], default='livecam', help="Mode")
parser.add_argument('--input', type=str, default='', help="Input image for detection")
args = parser.parse_args()
# Uno-card Detection (colour+shape) using the dataset & webcam – python script

# Detect the UNO given an image
def detect_uno(frame):
  # Find cards in webcam feed
  card_imgs, card_outlines = find_cards(frame)
  if len(card_outlines) > 0:
    cv.drawContours(frame, np.int0(card_outlines), -1, (0, 0, 255), 4)

  if len(card_imgs) > 0:
    # Normalize size for card from webcam feed
    for i in range(len(card_imgs)):
      card_imgs[i] = cv.resize(card_imgs[i], (CARD_WIDTH, CARD_HEIGHT), interpolation=cv.INTER_AREA)

    colors = []
    numbers = []

    # For each card, determine color and number
    for i in range(len(card_imgs)):
      # Determine image color
      dominant_color = find_color(card_imgs[i])

      # Pick closest in train data
      # Consider the Hue wrap around for RED for example,
      # say 255 is the reference value and the measured value is 0
      # , this will do a difference of 255 but actually it should be 1
      closest_idx = np.argmin([
        np.min([abs(dominant_color - c), abs((dominant_color+255) - c), abs(dominant_color - (c+255))])
        for c in dominant_colors])

      # Determine image number contour
      cnt = find_number_contour(card_imgs[i])

      # If no center contour, output unknown
      if cnt is None:
        predict_num = "UNKNOWN"
      else:
        # Pick closest match in contour
        match_idx = np.argmin([cv.matchShapes(cnt,c,1,0.0) for c in train_contours])
        predict_num = labels[match_idx][1]



      colors.append(labels[closest_idx][0])
      numbers.append(predict_num)

    for i in range(len(card_imgs)):
      x, y = np.int0(card_outlines[i][0])
      cv.putText(frame, colors[i] + " " + numbers[i], (x-10, y-10), cv.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), thickness = 2, lineType=cv.LINE_AA)


# Load dataset
images = []
labels = []

# Iterate through image files in dataset
fns = glob.glob(os.path.join(args.images, "*"))
for fn in fns:
  # Load image
  img = cv.imread(fn)
  images.append(img)

  # Parse image filename for label
  bname = os.path.basename(fn) # only get filename ( remove directory path )
  rname, _ = os.path.splitext(bname) # remove extension

  # Example names:
  #  * ZERO_B
  #  * TWO_B_DRAW
  #  * SKIP_Y
  # 
  # The second part is the color (R,G,B,Y)
  # rest is number (or special)
  name_parts = rname.split("_")
  color = name_parts[1]
  label = "_".join([name_parts[0]]+name_parts[2:])

  # Now color is R,G,B,Y
  # label is ZERO, TWO_DRAW, SKIP, ...
  labels.append([color, label])


# Save dimensions of train data, this will
# be useful later.
train_h, train_w = images[0].shape[:2]

# For each train image, find the card inside it
# and only keep the card image
for i in range(len(images)):
  card_imgs, _ = find_cards(images[i])
  if len(card_imgs) != 1:
    pass
    # Multiple or zero cards detected in train data image
    #print(f"Incorrect number of cards detected in {fns[i]}!")
  else:
    # Found one image, save it
    images[i] = card_imgs[0]


# Normalize each card image
for i in range(len(images)):
  images[i] = cv.resize(images[i], (CARD_WIDTH, CARD_HEIGHT), interpolation=cv.INTER_AREA)


# Find dominant color each card
dominant_colors = []

for i in range(len(images)):
  dominant_color = find_color(images[i])
  dominant_colors.append(dominant_color)

train_contours = []

# Find the contour of the number in the center
# for each card and save it.
for i in range(len(images)):
  cnt = find_number_contour(images[i])
  train_contours.append(cnt)


if args.mode == "livecam":
  # Open camera, pick the default source
  cap = cv.VideoCapture(args.camdevice)

  #------------------------------------
  # Camera process loop
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    detect_uno(frame)
    cv.imshow("UNO Card detection", frame)


    # Quit on escape
    if cv.waitKey(1) == 27:
      break

  cap.release()

elif args.mode == "staticimg":
  frame = cv.imread(args.input)
  detect_uno(frame)
  cv.imshow("UNO Card detection", frame)

  cv.waitKey()
