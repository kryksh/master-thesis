import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import dlib

def get_frontal_image(image):

    predictor_path='dlib_models/shape_predictor_68_face_landmarks.dat'
    model3D_path='frontalization_models/model3Ddlib.mat'
    eyemask_path='frontalization_models/eyemask.mat'

    model3D = frontalize.ThreeD_Model(model3D_path, 'model_dlib')
    eyemask = np.asarray(io.loadmat(eyemask_path)['eyemask'])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(image, 1)
    height , width , layers =  img.shape
    new_h = round(height/3)
    new_w = round(width/3)
    img = cv2.resize(img, (new_w, new_h))

    model3D = frontalize.ThreeD_Model('frontalization_models/model3Ddlib.mat', 'model_dlib')
    lmarks = feature_detection.get_landmarks(img, detector, predictor)
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    cv2.imwrite('frontal_image.png', frontal_sym)
