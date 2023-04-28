import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score




# Load the dataset
data = []
labels = []
for color in ['blue', 'green', 'red', 'yellow']:
    for i in range(0,12):
        folder_path = f"CW2_UNO/Dataset/{color}/{color}_{i}"
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (100, 100))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                hist = cv2.calcHist([img], [1, 2], None, [8, 8], [0, 256, 0, 256])
                
                hist = cv2.normalize(hist, hist).flatten()
                data.append(hist)

                labels.append(color)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test the model on live camera feed
vc = cv2.VideoCapture(0) 
while vc.isOpened():
    rval, frame = vc.read()    # read video frames again at each loop, as long as the stream is open
    
    # Predict the color and display it on the live feed
    img = cv2.resize(frame, (100, 100))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([img_lab], [1, 2], None, [8, 8], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    prediction = knn_model.predict([hist])[0]
    cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("stream", frame)
    
    key = cv2.waitKey(1)       # allows user intervention without stopping the stream (pause in ms)
    if key == 27:              # exit on ESC
        break
cv2.destroyWindow("stream")    # close image window upon exit
vc.release()
