# Mosaic-PS2

This repository contains our codes of 2nd Problem Statement of Mosiac'21 sponsored by Rapid AI
<br>

## Problem statement:
The Problem Statement was to detecting and recognizing the licence plate of a car. 
- The main part was to segmenting out characters from a Licenceplate and recognizing them.
- There was a bonus part of detecting the licence plate in real time video.
<br>

## Approach:
### Segmentation:
- For character segmentation we have used a Sobel based detector for detecting the edges of the plate.
- After detecting we have segmented out the plate and then used a four point transformation to change it to Birds eye view.
- After this we have segmented out the characters from the plateusing contour detection.
- For more accurate seperation of the characters we have used a **_minima based approach_**.

### Prediction:
- For predicting the letters we have trained a model for the characters.
- **confusion matrix**

### Video Processing:
- Firstly we have detected a car from a video using the YOLOv3 object detection model.
- Then we have tracked that car using OpenCV KCF Tracker until it goes out of field of view to avoid repeatation.
- Then we have detected the number plate of the car from the image taken using Haar Cascade detector.

<br>

## References:
- 
