<h1 align="center">Mosaic-PS2</h1>
<p align="center">
<h2 align = "center">Indian License Plate Recognition</h2>
<img width = 1000 height = 300 src = "imgs/plate.jpeg">
</p>

This **repository** contains our codes and approach for 2nd Round Problem Statement of Mosiac'21 sponsored by RapidAI.

## Problem statement:

The Problem Statement was to detecting and recognizing the licence plate of a car. 
- The main part was to segmenting out characters from a Licenceplate and recognizing them.
- There was a bonus part of detecting the licence plate in real time video.

For more details on PS, visit [Here](Mosaic'21_PS2.pdf)
## Approach:
### Segmentation:
- For character segmentation we have used a Sobel based detector for detecting the edges of the plate.
- After detecting we have segmented out the plate and then used a four point transformation to change it to Birds eye view.
- After this we have segmented out the characters from the plate using contour detection.
- For more accurate seperation of the characters from extreme noisy plates we have used a **_minima based approach_**.It calculates the frequency of horizontal projection then used the noise reduction using convolution with moving averages and cutting based on miminas. One downside of this appraoch is that it can detect false positives, which we are trying to rectify.

### Training and Prediction:
- For predicting the letters we have trained a model for the characters.
- The Confusion Matrix is shown here
- Can check notebook for training [here](train.ipynb)
<p align="center">
    <img height="512" width="512" src="imgs/cm.jpeg">
</p>

### Video Processing:
- Firstly we have detected a car from a video using the **YOLOv3 object detection model**.
- Then we have tracked that car using OpenCV KCF Tracker until it goes out of field of view to avoid repeatation.
- Then we have detected the number plate of the car from the image taken using Haar Cascade detector.
<p align="center">
    <img height="512" width="512" src="imgs/tracking.gif">
</p>
<p align="center">
    <img height="160" width="120" src="imgs/6.jpg">
    <img height="160" width="120" src="imgs/12.jpg">
    <img height="160" width="120" src="imgs/13.jpg">
    <img height="160" width="120" src="imgs/14.jpg">
    <img height="160" width="120" src="imgs/15.jpg">
</p>
<br>

## Testing:
- Install the requirements from the requirements.txt file.
- For segmenting out characters from a plate run the main.py file. It will ask for an image path whrere you provide the path.
- For tracking vehicles first download the weights and configuration file from [here](https://pjreddie.com/darknet/yolo/). Then run the tracking.py file, provide the video path.
<br>

<p align="center">
    <img height="160" width="1000" src="imgs/full.png">
    <b>OUR TESTING SNAPSHOT</b>
</p>

## References:
- [Four Point Transformation](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)
- [Object Detection by YOLOv3](https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420)
- [Object Tracking OpenCV](https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/)
- [Curve Smoothness](https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way)