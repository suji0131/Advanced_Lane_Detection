
# coding: utf-8

# In[1]:

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Perspective_Transform(object):
    #This class gives the perspective transformation of an image
    def __init__(self):
        src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680],
        ])

        dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    #warping the image
    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    #unwarping the image
    def unwarp(self, image):
        return cv2.warpPerspective(image, self.M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def __call__(self, image, unwarp=False):
        if unwarp:
            return self.unwarp(image)
        else:
            return self.warp(image)


# In[ ]:



