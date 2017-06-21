import argparse
import sys
from FeatureGen import*
import dlib
from skimage import io
import numpy
import cv2
from sklearn.externals import joblib
import os

class ExpressionDetector(object):
    """usage
        img = cv2.imread("/root/dl-data/github/face-expression-detect/t2.jpg")
        ed = ExpressionDetector()
        res = ed.predict_expression(img)
        """
    def __init__(self, landmark_path="shape_predictor_68_face_landmarks.dat",
                 train_data_pkl_path="traindata.pkl", pca_data_pkl_path="pcadata.pkl"):
        self.emotions={ 1:"Anger", 2:"Contempt", 3:"Disgust", 4:"Fear", 5:"Happy", 6:"Sadness", 7:"Surprise"}
        self.detector= dlib.get_frontal_face_detector()
        # load shape_predictor_68_face_landmarks.dat
        if not os.path.exists(landmark_path):
            print "Can not find " + landmark_path
        self.predictor= dlib.shape_predictor(landmark_path)
        # load traindata.pkl
        if not os.path.exists(train_data_pkl_path):
            print "Can not find " + train_data_pkl_path
        self.classify=joblib.load(train_data_pkl_path)
        # load pcadata.pkl
        if not os.path.exists(pca_data_pkl_path):
            print "Can not find " + pca_data_pkl_path
        self.pca=joblib.load(pca_data_pkl_path)
    
    def predict_expression(self, image):
        """predict human expression in the image
            Args:
            image (3-D ndarray)
            Return:
            emotion (str) : one of 'Anger','Contempt','Disgust','Fear','Happy','Sadness','Surprise'
            """
        dets=self.detector(image,1)
        if len(dets)==0:
            print "Unable to find any face."
            return
        shape=self.predictor(image,dets[0])
        
        landmarks=[]
        # get the point location
        for i in range(68):
            landmarks.append(shape.part(i).x)
            landmarks.append(shape.part(i).y)
        landmarks=numpy.array(landmarks)
        
        features=generateFeatures(landmarks)
        features= numpy.asarray(features)
        
        pca_features=self.pca.transform(features)
        
        emo_predicts=self.classify.predict(pca_features)
        emotion = self.emotions[int(emo_predicts[0])]
                return emotion
