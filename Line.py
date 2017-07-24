
# coding: utf-8

# In[1]:

import numpy as np
import collections

from Pipeline import *
from Perspective import *

class Line(object):
    # draws 2D polynomial approximation lines on the
    def __init__(self, x, y, height, width):
        self.height = height
        self.width = width
        self.recent_x_fitted = collections.deque(maxlen=5)
        self.recent_fits = collections.deque(maxlen=5)
        self.current_fit = None

        self.fit(x, y)

    def fit(self, x, y):
        if len(y) > 0 and (self.current_fit is None or np.max(y) - np.min(y) > self.height * 0.625):
            self.current_fit = np.polyfit(y, x, 2)
            self.recent_fits.append(self.current_fit)
            self.recent_x_fitted.append(x)

    def points(self):
        # For y points between 0 and height, we find the x-coordinates
        y_points = np.linspace(0, self.height - 1, self.height)
        best_fit = np.array(self.recent_fits).mean(axis=0)
        best_fit_x = best_fit[0] * y_points ** 2 + best_fit[1] * y_points + best_fit[2]
        return np.stack((best_fit_x, y_points)).astype(int).T

    def measure_curvature(self):
        points = self.points()
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        x = points[:, 0]
        y = points[:, 1]

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curve_radius = ((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5)                        / np.absolute(2 * fit_cr[0])
        return int(curve_radius)


# In[2]:


class laneTracker(object):
    """
    Creates a window object which can be used to map the points that belong to a lane edge. This window
    object returns a tuple of the points that form a rectangle
    """
    MARGIN = 100
    MIN_PIXELS = 50

    class window(object):
        def __init__(self, xmid, yLow, yHigh):

            """
            xmid : X coordinate of the center of rectangle window
            yTop : Top y coordinate of the rectangle
            yBottom : Bottom y coordinate of the rectangle
            margin : The stretch of the window from the midpoint towards the left and the right. Window width is 2*margin
            minpix : Minimum number of pixels we need to detect within a window to adjust the window location
            """
            self.xMid = xmid
            self.xMean = xmid
            self.yLow = yLow  # The y with a lower y value
            self.yHigh = yHigh  # The y with a higher y value
            self.xLeft = self.xMid - laneTracker.MARGIN
            self.xRight = self.xMid + laneTracker.MARGIN

        def find_pixel_indices(self, nonzero, x=None):

            if x is not None:
                self.xMid = x
                self.xLeft = self.xMid - laneTracker.MARGIN
                self.xRight = self.xMid + laneTracker.MARGIN

            nonzeroy = nonzero[0]
            nonzerox = nonzero[1]

            indices = ((nonzeroy >= self.yLow) & (nonzeroy < self.yHigh) &
                       (nonzerox >= self.xLeft) & (nonzerox < self.xRight)).nonzero()[0]

            if len(indices) > laneTracker.MIN_PIXELS:
                self.xMean = np.int(np.mean(nonzerox[indices]))
            else:
                self.xMean = self.xMid
            return indices

        def win_coordinates(self):

            # Returns the coordinates of the window as a tuple ((x1,y1),(x2,y2)), these are enough to define the rectangle
            return ((self.xLeft, self.yHigh), (self.xRight, self.yLow))

    def __init__(self, image):

        self.image = image
        self.num_windows = 9
        self.height, self.width, _ = image.shape
        self.window_height = np.int(self.height / self.num_windows)

        self.left_lane = None
        self.right_lane = None

        # Empty lists for left and right lane pixel indices
        self.left_lane_windows = []  # Store the window objects in a list
        self.right_lane_windows = []
        self.current_left_x = []
        self.current_left_y = []
        self.current_right_x = []
        self.current_right_y = []
        self.pipeline()

    def pipeline(self):

        image = gradWarpPipeline()(self.image)

        # Histogram of the bottom half of the image
        histogram = np.sum(image[self.height // 2:, :], axis=0)

        # To find the peaks of the histogram. These peaks tell us where the lane edges are
        midpoint = np.int(histogram.shape[0] / 2)
        left_x = np.argmax(histogram[:midpoint])
        right_x = np.argmax(histogram[midpoint:]) + midpoint

        left_lane_indices = []
        right_lane_indices = []

        # Get nonzero pixels in the image. The nonzero() return a tuple of x, y points
        # that have nonzero value
        nonzero = image.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        # Stepping through the windows one by one

        for window in range(self.num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window + 1) * self.window_height
            win_y_high = image.shape[0] - window * self.window_height

            left_window = laneTracker.window(left_x, win_y_low, win_y_high)
            right_window = laneTracker.window(right_x, win_y_low, win_y_high)

            # A list consisting of window objects
            self.left_lane_windows.append(left_window)
            self.right_lane_windows.append(right_window)

            # Get nonzero pixel indices within the window
            left_nonzero_indices = left_window.find_pixel_indices(nonzero)
            right_nonzero_indices = right_window.find_pixel_indices(nonzero)

            # Append the window nonzero pixel indices to the lists
            left_lane_indices.append(left_nonzero_indices)
            right_lane_indices.append(right_nonzero_indices)

            left_x = left_window.xMid
            right_x = right_window.xMid

        # Concatenate all the windows nonzero pixel indices
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        self.left_lane = Line(nonzerox[left_lane_indices],
                              nonzeroy[left_lane_indices],
                              self.height,
                              self.width)

        self.right_lane = Line(nonzerox[right_lane_indices],
                               nonzeroy[right_lane_indices],
                               self.height,
                               self.width)

    def find_lanes(self, image):
        # pipeline returns bird eye view perspective image, after applying sobel gradient
        # and color thresholds. The return image has pixel values of 0 and 1
        bird_eye_view = gradWarpPipeline()(image)
        # find all the nonzero pixels for the left lane and fit a polynomial curve
        self.find_left_lane(bird_eye_view)
        # find all the nonzero pixels for the right lane and fit a polynomial curve
        self.find_right_lane(bird_eye_view)

        self.draw_text(image)

        image = self.draw_lanes(image, unwarp=True)
        return image

    def draw_text(self, image):
        self.put_text(image, 'Radius of curvature:  {} m'.format(self.measure_curvature()), 20, 80)
        vehicle_pos = self.vehicle_position()
        self.put_text(image, 'Estimated offset from lane center:    {:.1f} m'.format(vehicle_pos), 500, 80)

    def find_left_lane(self, image):
        self.current_left_x, self.current_left_y =             self.find_lane(image, self.left_lane_windows)
        self.left_lane.fit(self.current_left_x, self.current_left_y)

    def find_right_lane(self, image):
        self.current_right_x, self.current_right_y =             self.find_lane(image, self.right_lane_windows)
        self.right_lane.fit(self.current_right_x, self.current_right_y)

    def find_lane(self, image, windows):
        indices = []
        # Get nonzero pixels in the image. The nonzero() return a tuple of x, y, z points
        nonzero = image.nonzero()
        x = None
        for window in windows:
            indices.append(window.find_pixel_indices(nonzero, x))
            x = window.xMean
        indices = np.concatenate(indices)
        return nonzero[1][indices], nonzero[0][indices]

    def draw_lanes(self, frame, unwarp=False):
        image = np.zeros_like(frame).astype(np.uint8)
        points = np.vstack((self.left_lane.points(), np.flipud(self.right_lane.points())))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(image, [points], (0, 255, 0))
        if unwarp:
            image = Perspective_Transform().unwarp(image)
        # Combine the result with the original image
        return cv2.addWeighted(frame, 1, image, 0.3, 0)

    def put_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def measure_curvature(self):
        return int(np.average([self.left_lane.measure_curvature(),
                               self.right_lane.measure_curvature()]))

    def vehicle_position(self):
        #function calculates the vehicle position
        left_lane_points = self.left_lane.points()
        right_lane_points = self.right_lane.points()
        lane_middle = int((right_lane_points[right_lane_points[600,1]][0] - left_lane_points[left_lane_points[600,1]][0])/2.) + left_lane_points[left_lane_points[600,1]][0]


        if (lane_middle - 640 > 0):
            leng = 3.66/2
            mag = ((lane_middle - 640)/640.*leng)
            head = ("Right", mag)
            return mag

        else:
            leng = 3.66 / 2
            mag = ((lane_middle - 640)/640*leng)* -1
            head = ("Left", mag)
            return mag


# In[ ]:



