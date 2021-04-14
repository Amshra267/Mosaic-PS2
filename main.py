import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segmentar import extraction
from tensorflow.keras.models import load_model

mapping= {

    0 : "0", 1 : "1", 2 : "2", 3 : "3", 4 : "4", 
    5 : "5", 6 : "6", 7 : "7", 8 : "8", 9 : "9", 
    10 : "A", 11 : "B", 12 : "C", 13 : "D", 14 : "E", 
    15 : "F", 16 : "G", 17 : "H", 18 : "I", 19 : "J",
    20 : "K", 21 : "L", 22 :"M", 23: "N", 24 : "O",
    25 : "P", 26 : "Q", 27 : "R", 28 : "S", 29 : "T",
    30 : "U", 31 : "V", 32 : "W", 33 : "X", 34 : "Y", 35: "Z"

}

##-------loading model with weights---
model = load_model("model.h5")


if __name__=="__main__":
    print("Enter the path of input image")

    while True:
        path = input()
        if not os.path.exists(path):
            print("File not exist, please enter correct path once more")
            continue
        break

    img = cv2.imread(path)
 #   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## resizing image with proper aspect ratio------------
    image_gen  = extraction(img) ## GENERATOR CALLED
    i=0
    word = []
    while True:
        try:
            image = next(image_gen)
            print(image.shape)
            # image = image.reshape(-1,28,28,1).astype("float32")/255
    
            # prediction = model(image)
            # cls = np.argmax(prediction, axis = 1)
            # print(cls)
            # print(prediction[0, cls])
            
            # word.append(mapping[cls[0]])
            cv2.imshow("imgs" + str(i), image)
            i+=1
        except Exception as e:
            print(e)
            break

    #------------Resizing_finished--------------
    #print(" ".join(word))

    #cv2.imshow("extract", img)
     
    if cv2.waitKey(0)&0Xff ==27:
        cv2.destroyAllWindows()