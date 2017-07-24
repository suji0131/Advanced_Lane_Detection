# Advanced Lane Detection

## Camera Calibration
Camera calibration (done to eliminate the distortion induced by lens of a camera) is done in CameraCalibration.py using the 
chess board images in camera_cal folder. OpenCV function findChessboardCorners is used to find the corners of a chess board and
calibratecamera to find camera matrix and distortion coefficients. 

![CameraCalibration](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/CameraCalibration.png)

![Distortion Correction for RealImage](https://github.com/suji0131/Advanced_Lane_Detection/blob/master/output_images/DistortionCorrectionRealImage.png)
