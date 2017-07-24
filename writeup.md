# Advanced Lane Detection

## Camera Calibration
Camera calibration (done to eliminate the distortion induced by lens of a camera) is done in CameraCalibration.py using the 
chess board images in camera_cal folder. OpenCV function findChessboardCorners is used to find the corners of a chess board and
calibratecamera to find camera matrix and distortion coefficients. 

![CameraCalibration](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/CameraCalibration.png)

![Distortion Correction for RealImage](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/DistortionCorrectionRealImage.png)

## Pipeline
Pipeline.py describes the pipeline I used to detect the lane line. A combination of color and gradient thresholds are used to generate a binary image. Detailed steps can be found in the Final_adv_lane_detect.ipynb. But first image is converted to HLS color channel. Final output image looks like below:

![Thresholding](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/Thresholding.png)

## Perspective Transformation
Class written in Perspective.py gives the perspective transformation of a given image. warpPerspective function from cv2 is used. Source and destination points used are in the same file.

![PerspectiveTransformation](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/PerspectiveT.png)

## Lane Detection and Lane Curvature
Both of these tasks can be found in Line.py file. A histogram is used to identify the pixels which form part of the lane. Adding the pixel values along each column in the image, the peaks in the histogram are found and these peaks are good indicators of the x-position of the base of the lane lines. 

![Histogram](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/Histogram.png)

These points are used as starting points to search for the lines. From here on, I used a sliding window placed around line centers to form the new windows up till the top of the frame. After that, I scanned each window to collect the non-zero pixels within window bounds. Then a second order polynomial can be fit to the collected points.
![SlideWindow](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/SlidingWindow.png)

To determine the Lane curvature and vehicle position below equations are used:
![LaneCurve](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/LaneCurveEq.png)

Final processed image looks like below:
![Final Image](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/FinalImg.png)

Here is the link to my [final video](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/project_video_final.mp4)

I wrote most of my work in classes but in hindsight I should have used just functions as processing time required for a minute of video is nearly four and half minutes, which is impractical as a self driving car has to identify lanes, position of car and take a decision within a few fraction of a second. For future work I will try to speed up my pipeline.
