# CW2_UNO-CARDS-DETECTOR-PDE4434

The objective of this coursework is to develop a software that can identify UNO cards from either an image or a frame by utilizing information from a dataset or a standard camera, such as a webcam.

#
## Demo

You can watch a demo of this project in action on my YouTube channel:[https://youtu.be/FOBK394PFFI] In the video, I demonstrate how the software can identify UNO cards from both images and live camera feeds using computer vision techniques. I also provide a step-by-step guide on how to run the program with the provided dataset or your own images.

Please visit My GitHub profile to access the link. [https://github.com/mdileepkumar438/CW2_UNO]

#



## The program can be run with the following command-line arguments:
```
Main file 
UNO_card_detector.py using the dataset & webcam 
```


* To run using Dataset images:

`Example input names:`

`1.   ZERO_B`

`2.   TWO_B_DRAW`

`3.   SKIP_Y`

```
python3 uUNO_card_detector.py --images Main_Dataset --input image.jpg --mode staticimg
  
```

* To run script using the live cam (webcam): `Default camdevice(0)`

```
python3 UNO_card_detector.py --images Main_Dataset --camdevice 0 --mode livecam
```

## Notes:

1. "Main_Dataset" is the name of the folder that contains the data set “UNO cards Images”

2. Ensure that the webcam/camera is set up with a black/dark background when conducting tests.

* Make sure to have the below libraries:

```
glob ---  'pip install glob2'
cv2  ---  'pip install opencv-python'
numpy --- 'pip install numpy'
argparse --- 'pip install argparse'
matplotli --- 'pip install matplotlib'
```
#

* This image provides a glimpse of the physical project utilizing a webcam.

![WhatsApp Image 2023-04-28 at 10 52 11 PM](https://user-images.githubusercontent.com/102908088/235231974-72d0a3b6-1479-489e-95cb-e80ee1648458.jpeg)

![WhatsApp Image 2023-04-28 at 10 52 12 PM](https://user-images.githubusercontent.com/102908088/235232012-8586b651-8e9d-4aa4-8235-2c462fb62ab0.jpeg)


<img width="950" alt="Screenshot 2023-04-28 at 10 22 28 PM" src="https://user-images.githubusercontent.com/102908088/235239292-1025874c-77da-42b1-aa67-992ed5a1dd9e.png">

* Using Static Image 

<img width="413" alt="Screenshot 2023-04-28 at 11 16 12 PM" src="https://user-images.githubusercontent.com/102908088/235240044-43a99820-eede-4b56-96ce-600acb1f5478.png">





# Workflow of the Project
## 1. Color_estimation.py

This is a simple Python script that uses OpenCV and scikit-learn libraries to build a K-nearest neighbor (KNN) classifier model for identifying four colors - blue, green, red, and yellow. The dataset used for training the model contains 48 images of each color (12 images per shade). The model is then tested on a live camera feed to identify the color of the object in the frame.

## 2. color_Detection.ipynb
1. The dataset images are loaded.
2. The card is cropped out of each image in the dataset.
3. The dataset is split into training and testing sets in an 80/20 ratio.
* For the training set, color detection is performed as follows:

    The image is resized to a small size using the NEAREST interpolation method.

    K-means clustering (K=2) is applied to identify the white color and the color of the uno card.

    The color farthest from white is chosen as the uno color.

    The color is then transformed into the HSV color space, and the hue value is selected as the dominant color.


* For the test set, the following steps are carried out:

    The dominant color is determined as described above.

    The dominant color is compared to the dominant color in the training set.

    The closest hue value in the training set is selected.
    It is ensured that the color label in the training and test sets match.

## 3. Color_shape_estimator.ipynb

This script uses computer vision techniques to detect and recognize UNO cards in real-time using a webcam. The script performs the following steps:

* Loads a dataset of UNO card images and their corresponding labels.

* Finds the contour of the number in the center of each card image in the dataset.

* Normalizes each card image in the dataset to a fixed size.
* Finds the dominant color in each card image in the dataset.
* Opens a webcam and starts reading frames.
* Detects UNO cards in each frame and normalizes them to the same size as in step 3.
* Determines the color and number of each card in the webcam frame.
* Draws the color and number of each card on the frame.

To run the script, ensure that the necessary libraries are installed: OpenCV, Numpy, and OS. Also, ensure that a webcam is connected to the computer. The script will automatically use the default webcam as the video source. The UNO dataset is expected to be in the "./Main_Dataset/" folder in the script's directory.

## 3. UNO_card_detector.py using the dataset & webcam – python script

This program can recognize UNO card color and number using a webcam or static images. It can detect the UNO cards present in the image and normalize their size. Then it determines the color and number on each card and outputs them on the image. The program uses OpenCV, numpy, and argparse libraries. The program's main functions are:

- detect_uno(frame) function to detect and recognize UNO cards from input image.
- find_cards(frame) function to find all the cards present in the image.
- find_number_contour(image) function to find the contour of the number or special character in the center of the card.
- find_color(image) function to find the dominant color in the image.

The program uses a dataset of UNO cards to recognize them. 
It loads the dataset images and their labels and uses them to determine the color and number of the cards.



Usage 

After combining (colour+number) I have added the following arguments:

a- ('--images', help="Input CARDS")

b- ('--camdevice', type=int, default=0, help="Camera number")

c- ('--mode', choices=['livecam', 'staticimg'], default='livecam', help="Mode")

d- ('--input', type=str, default='', help="Input image for detection")

#
#
#

References:

* here are the references for the libraries used in the code:

1. OpenCV - https://opencv.org/
2. Numpy - https://numpy.org/
3. argparse - https://docs.python.org/3/library/argparse.html
4. glob - https://docs.python.org/3/library/glob.html
5. os - https://docs.python.org/3/library/os.html


Here are some additional links related to UNO card detection:

1. UNO card game - https://en.wikipedia.org/wiki/Uno_(card_game)
2. Object detection with OpenCV - https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
3. PLaying Cards Detection - https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector    
