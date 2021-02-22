# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 22:44:39 2021

@author: Taniya
"""
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

from flask import Flask
from flask import request
from flask import render_template



app=Flask(__name__)
UPLOAD_FOLDER='static'


def predict(image1):
    face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('EmotionDetectionModel.h5')
    class_labels=['Angry','Happy','Neutral','Sad','Surprised']
    frame=cv2.imread(image1)
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    face=faces[0]
    x,y,w,h=face
    
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    
    if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
    
                preds=classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
         
    return label

@app.route('/', methods=['GET', 'POST'])

def upload_predict():
    if request.method=="POST":
        image_file=request.files["image"]
        if image_file:
            image_location=os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)
            print(image_location)

            pred=predict(image_location)
            print(pred)
            return render_template('index.html',prediction=pred)
    return render_template('index.html',prediction="Let's find it!")

if __name__ == "__main__":
  app.run(port=5000)
    
    