import cv2
import dlib
import shelve
import numpy as np 
import copy
import os 
import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib


def get_def_pts(image, predictor_path='dlib_models/shape_predictor_68_face_landmarks.dat', 
                  model3D_path='frontalization_models/model3Ddlib.mat', 
                  eyemask_path='frontalization_models/eyemask.mat', resize=False):

  list_def_face = []
  
  for f_name in sorted(os.listdir('data')):
    if f_name.startswith('def_face'):
        list_def_face.append(f_name)
    
  points_list = []
  print(image)
  img = cv2.imread(image, 1)

  height , width , layers =  img.shape
  if resize==True:
      new_h=round(height/3)
      new_w=round(width/3)
      img = cv2.resize(img, (new_w, new_h))
      
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(predictor_path)
  
  lmarks = feature_detection.get_landmarks(img, detector, predictor)
  points_list=[[x ,y] for x, y in lmarks[0]]
  
  for num, file in enumerate(list_def_face):
    with shelve.open('extracted_data/default_points_{}.txt'.format(num), 'c') as db:
        for i, x in enumerate(points_list):
            db['default_face'] = points_list
  
  return(points_list)

def extract_def_pts():
    for i in range(sum(f_name.startswith('def_face') for f_name in os.listdir('data'))):
        get_def_pts('data/def_face_{}.png'.format(i))

if __name__ == '__main__':
    extract_def_pts()

