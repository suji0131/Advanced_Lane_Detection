
# coding: utf-8

# In[ ]:

import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

class CameraCalibrate():
    def __init__(self, img_folder, corners=(8,6)):
        
        #reading list of calibration images
        self.images = glob.glob(img_folder+'calibration*.jpg')
        self.corners = corners
        
        #parameters
        self.camera_matrix = None
        self.distortion_coeff = None
        self.calibrate()
        
    def calibrate(self):
        objpts = [] #real world space points
        imgpts = [] #image space points
        
        #initial object points
        objp = np.zeros([self.corners[0]*self.corners[1],3], np.float32)
        #creating the x,y coordinates for the corners
        objp[:, :2] = np.mgrid[0:self.corners[0], 0:self.corners[1]].T.reshape(-1,2)
        
        for name in self.images:
            print(name)
            #read in the image
            img = mpimg.imread(name)
            
            #convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            #finding chessboard corners
            found, corners = cv2.findChessboardCorners(gray,(8,6),None)
            
            if found == True:
                imgpts.append(corners)
                objpts.append(objp)
                
        found, self.camera_matrix, self.distortion_coeff, _, _ = cv2.calibrateCamera(objpts, imgpts, gray.shape,None, None)
    def undistort(self,image):
        #fn for undistorting the image
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeff, None, self.camera_matrix)
    
    def __call__(self,image):
        return self.undistort(image)

