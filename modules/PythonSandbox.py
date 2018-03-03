import libjevois as jevois
import cv2
import numpy as np

## Using OpenCV to detect field elements
#
# @author Cole Hersowitz
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois PythonSandbox
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PythonSandbox:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("detector", 100, jevois.LOG_INFO)
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # setup query image (training)
        '''query_img = cv2.imread('test.png', 0)

        # create surf object
        surf = cv2.SURF(400)

        # find keypoints and descriptors for training and actual image
        kps_target, desc_target = surf.detectAndCompute(query_img, None)
        kps_actual, desc_actual = surf.detectAndCompute(inimg, None)

        # see if training and actual match
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_target, desc_actual, k=2)

        # ratio test for match 
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # create output img
        outimg = cv2.drawMatchesKnn(query_img, kps_target, inimg, kps_actual, good, flags=2)'''
        marker = find_marker(cv2.imread('test.png', 0))
        focalLength = (marker[1][0] * 24.0) / 5.0
        inches = distanceToCam(5.0, focalLength, marker[1][0])

        box = np.int0(cv2.cv.BoxPoints(marker))
        cv2.drawContours(inimg, [box], -1, (0, 255, 0), 2)
        cv2.imshow("image", inimg)
        cv2.waitKey(0)
            
        # Write a title:
        cv2.putText(outimg, "6072 Field Object Detect", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),
                    1, cv2.LINE_AA)
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Convert our BGR output image to video output format and send to host over USB. If your output image is not
        # BGR, you can use sendCvGRAY(), sendCvRGB(), or sendCvRGBA() as appropriate:
        outframe.sendCvBGR(outimg)

    def get_training_images():
        imgs = []
        for img in open('dir_here'): #iterate through all images in training directory
            imgs.append(img)
        return imgs

    def get_surf_points():
        kpsAndDesc = []
        for img in get_training_images():
            cv_img = cv2.imread(image, 0)
            kp, des = surf.detectAndCompute(cv_img, None)
            kpsAndDesc.append([kp, des])
        return kpsAndDesc

    def find_matches(desc_actual):
        bf = cv2.BFMatcher()
        matches = []
        surf_pts = self.get_surf_points()
        for pt in surf_pts:
            matches.append(bf.knnMatch(pt[1], desc_actual))
        return matches 

    def apply_ratio_test(matches):
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return good 

    # tutorial ref: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    def find_marker(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        # find the contours in the image
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key = cv2.contourArea)

        return cv2.minAreaRect(c)

    def distanceToCam(knownWidth, focalLength, perWidth):
        return (knownWidth * focalLength) / perWidth;