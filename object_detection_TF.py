import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

import cv2

class image_identifier:

  __hub_model = ""

  def __init__(self):
    image_identifier.gpu_config(self)
    self.__hub_model = image_identifier.load_model()

  def gpu_config(self):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

  def load_model():
    tf.get_logger().setLevel('ERROR')
    
    ALL_MODELS = {'CenterNet HourGlass104 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1'}

    model_display_name = 'CenterNet HourGlass104 Keypoints 512x512'
    model_handle = ALL_MODELS[model_display_name]

    print('Selected model:'+ model_display_name)
    print('Model Handle at TensorFlow Hub: {}'.format(model_handle))

    print('loading model...')
    hub_model = hub.load(model_handle)
    print('model loaded!')

    return hub_model
  
  def object_detection(self, image_ocv):

    #Convierte de opencv a PIL
    img = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img)
    (im_width, im_height) = image_pil.size
    image_np = np.array(image_pil.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),(0, 2),(1, 3),(2, 4),(0, 5),(0, 6),(5, 7),(7, 9),(6, 8),(8, 10),(5, 6),(5, 11),(6, 12),(11, 12),(11, 13),(13, 15),(12, 14),(14, 16)]

    PATH_TO_LABELS = '/home/roberott/Desktop/prueba/models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # running inference
    results = self.__hub_model(image_np)

    # different object detection models have additional results
    # all of them are explained in the documentation
    result = {key:value.numpy() for key,value in results.items()}
    #print(result.keys())

    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
      keypoints = result['detection_keypoints'][0]
      keypoint_scores = result['detection_keypoint_scores'][0]
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections[0],
          result['detection_boxes'][0],
          (result['detection_classes'][0] + label_id_offset).astype(int),
          result['detection_scores'][0],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False, # Si TRUE, solamente detecta objetos pero no dice que son.
          keypoints=keypoints,
          keypoint_scores=keypoint_scores,
          keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    opencv_image = cv2.cvtColor(image_np_with_detections[0], cv2.COLOR_RGB2BGR)

    return opencv_image, keypoints, keypoint_scores

if __name__ == "__main__":
  #o = image_identifier()
  #o.object_detection()
  pass