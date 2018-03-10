import libjevois as jevois
import cv2
import numpy as np

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is here for you to experiment with Python OpenCV on JeVois.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an output BGR image named 'outimg'.
# We finally add some text drawings to outimg and send it to host over USB.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois PythonSandbox
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class ScaleDetect:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("ScaleDetect", 100, jevois.LOG_INFO)
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        inimg2 = inimg.copy()

        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # Detect edges using the Laplacian algorithm from OpenCV:
        #
        # Replace the line below by your own code! See for example
        # - http://docs.opencv.org/trunk/d4/d13/tutorial_py_filtering.html
        # - http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
        # - http://docs.opencv.org/trunk/d5/d0f/tutorial_py_gradients.html
        # - http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
        #
        # and so on. When they do "img = cv2.imread('name.jpg', 0)" in these tutorials, the last 0 means they want a
        # gray image, so you should use getCvGRAY() above in these cases. When they do not specify a final 0 in imread()
        # then usually they assume color and you should use getCvBGR() here.
        #
        # The simplest you could try is:
        #    outimg = inimg
        # which will make a simple copy of the input image to output.
        #outimg = cv2.Laplacian(inimg, -1, ksize=3, scale=0.25, delta=127)

        '''rgb_boundary = [
            ([204, 255, 255], [0, 102, 204]),
            ([204, 255, 255], [0, 102, 204]),
            ([153, 31, 0], [255, 71, 26])
        ]

        lower_rgb = None
        upper_rgb = None
        for (lower, upper) in rgb_boundary:
            lower_rgb = np.array(lower, dtype="uint8")
            upper_rgb = np.array(upper, dtype="uint8")'''
        hsv1 = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(inimg.copy(), cv2.COLOR_BGR2HSV)

        lower_red = np.array([30,150,50])
        upper_red = np.array([255,255,180])

        # purple identifies the center pivot 
        lower_purple = np.array([232, 232, 232])
        upper_purple = np.array([255, 255, 255])
 
        # Here we are defining range of bluecolor in HSV
        # This creates a mask of blue coloured 
        # objects found in the frame.
        red_mask = cv2.inRange(hsv1, lower_red, upper_red)
        blue_mask = cv2.inRange(hsv2, lower_purple, upper_purple)

        mask = red_mask 

        # clean up the mask a bit
        '''kernelOpen = np.ones((5,5))
        kernelClose = np.ones((20, 20))

        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
        maskFinal = maskClose'''
        #conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        outimg = cv2.bitwise_and(inimg, inimg, mask=mask)
        outimg2 = cv2.bitwise_and(inimg2, inimg2, mask=blue_mask)
        #outfinal = cv2.bitwise_or(outimg, outimg2)

        outfinal = cv2.add(outimg, outimg2)

        ## find the contours ###
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts2 = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        pt1 = None
        pt2 = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # make sure a min size is met 
            if radius > 20:
                pt1 = center
                cv2.circle(outfinal, center, 5, (0, 0, 255), -1)

        if len(cnts2) > 0:
            c2 = max(cnts2, key=cv2.contourArea)
            ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
            M2 = cv2.moments(c2)
            center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

            if radius2 > 20:
                pt2 = center2
                cv2.circle(outfinal, center2, 5, (255, 0, 255), -1)

        vert_dist = 0.1
        if pt2 != None and pt1 != None:
            vert_dist = pt1[1] - pt2[1]
            horizontal_dist = pt1[0] - pt2[0]
        ## attempt to find angle ###
        #moments1 = cv2.moments(outimg.resize(480, 640), 0)
        #moments2 = cv2.moments(outimg, 0)
        #area1 = moments1['m00']
        #area2 = moments2['m00']

                
        # Write a title:
        cv2.putText(outfinal, "JeVois Recognize Scale {}, {}, {}".format(vert_dist, pt1, pt2), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),
                    1, cv2.LINE_AA)
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
        cv2.putText(outfinal, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Convert our BGR output image to video output format and send to host over USB. If your output image is not
        # BGR, you can use sendCvGRAY(), sendCvRGB(), or sendCvRGBA() as appropriate:
        #all_output = np.dstack((outimg, outimg2))
        #outframe.send(outfinal)
        outframe.sendCvBGR(outfinal)
        
        # send data over serial usb
        # need to make sure setpar serout is enabled for this
        #data = {"message": "test"}
        #json_data = json.dumps(data)
        jevois.sendSerial("test22")
