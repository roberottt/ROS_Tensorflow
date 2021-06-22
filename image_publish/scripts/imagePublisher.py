#!/usr/bin/env python3

# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

import sys
sys.path.insert(1, '/home/roberott/Desktop/prueba')

import object_detection_TF
from models import *
 
VERBOSE=False

class image_feature:

    def __init__(self):
        
        self.ob = object_detection_TF.image_identifier()
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=10)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/image/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)
        
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        #### Feature detectors using CV2 #### 
        # "","Grid","Pyramid" + 
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
        method = "GridFAST"

        processed_image, keypoints, keypoints_scores= self.ob.object_detection(image_np) # Pasamos el frame por TF

        # Para guardarnos los keypoints de los objetos detectados
        """
        f = open("/home/roberott/Desktop/keypoints.txt", "a")
        for row in keypoints:
            np.savetxt(f, row)
        f.close()
                
        f = open("/home/roberott/Desktop/keypoints_scores.txt", "a")
        for row in keypoints_scores:
            np.savetxt(f, row)
        f.close()
        """
        cv2.imshow('cv_img', processed_image)
        cv2.waitKey(2)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', processed_image)[1]).tobytes()
        # Publish new image
        self.image_pub.publish(msg)



def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature', anonymous=True)
    ic = image_feature()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
