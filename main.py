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


def convention(predictions, length):
    """
    Function for implementimg the convention of Indian plates
    """
    cls = np.argmax(predictions, axis = 1)
    for i in [0,1]: # first 2 district code must represent letters
        if cls[i]==0:
            cls[i]= 24
        # else:
        #     print(np.argmax(predictions[i,10:]))
        #     cls[i] = 10+np.argmax(predictions[i,10:])
    for i in [-4,-3,-2,-1]: # last 4 numbers
        if cls[i]== 24:
            cls[i]=0
        # else:
        #     cls[i] = np.argmax(predictions[i,:10])
    diff = length-6
    for i in range(2, 2+diff):
        if i in [2,3]: # registration no.
            if cls[i]==24:
                cls[i]=0
            # else:
            #     cls[i] = np.argmax(predictions[i,:10])
    word = [mapping[predict] for predict in cls]
    return word

if __name__=="__main__":
    print("Enter the path of input image")

    while True:
        path = input()
        if not os.path.exists(path):
            print("File not exist, please enter correct path once more")
            continue
        break

 
    ## resizing image with proper aspect ratio------------
    imgs  = extraction(path) ## GENERATOR CALLED
    imgs = np.asarray(imgs)
    imgs = imgs.reshape(-1,64,64,1).astype("float32")/255
    if len(imgs)<6:
        predictions = model(imgs)
        cls = np.argmax(predictions, axis = 1)
        print(cls)
        word = [mapping[classes] for classes in cls]
    elif len(imgs)<8:
        predictions = model(imgs)
        word = convention(predictions, len(predictions))
    else:
        predictions = model(imgs)
        cls = np.argmax(predictions, axis = 1)
        word = convention(predictions, len(predictions))
        
    # print(prediction)
        
    for i in range(len(imgs)):
        cv2.imshow("imgs" + str(i), imgs[i,:,:])
        


    #------------Resizing_finished--------------
    print("".join(word))

    #cv2.imshow("extract", img)
     
    if cv2.waitKey(0)&0Xff ==27:
        cv2.destroyAllWindows()