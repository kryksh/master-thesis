import cv2
import dlib
from imutils.video import VideoStream
from imutils import face_utils
import shelve
import scipy.misc as sm
import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
import time 
from scipy import ndimage
import copy 
import os
import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io

def get_video_points(video_file, predictor_path='dlib_models/shape_predictor_68_face_landmarks.dat', 
                  model3D_path='frontalization_models/model3Ddlib.mat', 
                  eyemask_path='frontalization_models/eyemask.mat', 
                  mode='face', resize=False):
    
    time_list=[]
    points_list=[]
    points_list_sym=[]
    points_list_raw=[]
    
    model3D = frontalize.ThreeD_Model(model3D_path, 'model_dlib')
    eyemask = np.asarray(io.loadmat(eyemask_path)['eyemask'])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)    
    print('fps=', fps)
    
    # width = 320 
    # height = 320
    
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video_sym = cv2.VideoWriter('video_sym.avi', fourcc, fps, (width, height))
    # video_raw = cv2.VideoWriter('video_raw.avi', fourcc, fps, (width, height))
    
    while True:
        try:
            
            ret, frame = cap.read()
            if not ret:
              print("no ret")
              break

            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            print(msec/1000)
            
            img = frame
            iheight , iwidth , layers =  img.shape
            
            
            
            #resize
            if resize==True:
                new_h = round(iheight/3)
                new_w = round(iwidth/3)
                img = cv2.resize(img, (new_w, new_h))
                
            lmarks = feature_detection.get_landmarks(img, detector, predictor)
            
            if len(lmarks) != 1:
              continue
            
            proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
            
            frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
            
            time_list.append(msec)
            
            new_img_sym = np.round(frontal_sym).astype(np.uint8)
            new_img_raw = np.round(frontal_raw).astype(np.uint8)
            temp_pts=[]
            lmarks_sym = feature_detection.get_landmarks(new_img_sym, detector, predictor)
            temp_pts=[[x ,y] for x, y in lmarks_sym[0]]

            if mode == 'mouth':
                temp_pts = temp_pts[33:34] + temp_pts[48:]

            points_list_sym.append(temp_pts)
            
            # video_sym.write(new_img_sym)
            # video_raw.write(new_img_raw)
            
        except ValueError as e:
            print(e)
            break
    
    cap.release()
    points_list = list(zip(time_list, points_list_sym))
    
    
    with shelve.open('extracted_data/' + video_file.split('/')[-1].split('.')[0] + '.txt', 'c') as db:
        for i, x in enumerate(points_list):
            db[str(i)] = x    

    return(points_list)                       
    
    # video_sym.release() 
    # video_raw.release() 
    # cv2.destroyAllWindows()


def extract_points():
    for i in range(sum(f_name.startswith('video') for f_name in os.listdir('data'))):
        get_video_points('data/video_{}.mp4'.format(i))

if __name__ == '__main__':
    extract_points()