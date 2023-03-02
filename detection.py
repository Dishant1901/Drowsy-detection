import cv2
import os
from tensorflow.python.keras.models import load_model
import numpy as np
from pygame import mixer 
import time

mixer.init()
sound = mixer.sound('alarm.wav')

face = cv2.CascadeClassifier('harr cascade files\haarcascade_frontalface_alt.xml')
left_eye = cv2.CascadeClassifier('harr cascade files\haarcascade_lefteye_2splits.xml')
right_eye = cv2.CascadeClassifier('harr cascade files\haarcascade_righteye_2splits.xml')

# label to assinnged to state of eyes
label =[' closed','open']

# model which is yet to be created
# model = load_model('model/cnncat2.h5')

path = os.getcwd()

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN

count =0
reading = 0

# find out what does this and change its name
thicc =2
left_pred=[99]
right_pred=[99]


# reading the face and detecting eyes

while(1):

    ret,frame = cap.read()
    height ,width = frame.shape[:2]

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiscale(gray,minNeighbors=5,scaleFactor=1.1,miniSize=(25,25))
    LeftEye= left_eye.detectMultiScale(gray)
    RightEye=right_eye.detectMultiscale(gray)

    cv2.rectangle(frame,(0,height-50) , (200,height),(0,0,0), thickness=cv2.FILLED)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame , (x,y) , (x+w,y+h) , (100,100,100),1)

        # using model to pr3dict right eye 
    for(x,y,w,h) in RightEye:

        r_Eye = frame[y:y+h,x:x+w]
        count+=1

        r_Eye=cv2.cvtColor(r_Eye,cv2.COLOR_BGR2GRAY)

        # resize this if training model metricesis also changed
        r_Eye = cv2.resize(r_Eye,(24,24))
        r_Eye = r_Eye/255
        r_Eye= r_Eye.reshape(24,24,-1)
        r_Eye=np.expand_dims(r_Eye,axis=0)

        right_pred = model.predict_classes(r_Eye)

        # wtf does this conditions do???
        if(right_pred[0]==1):
            label= 'open'
        if(right_pred [0]==0):
            label = 'closed'
            break


        # using the model to predict for left eye
    for (x,y,w,h)  in LeftEye:
        l_Eye = frame[y:y+h,x:x+w]
        count+=1
        l_Eye = cv2.cvtColor(l_Eye,cv2.COLOR_BGR2GRAY)

            #  resize this if training model metricesis also changed
        l_Eye=cv2.resize(l_Eye,(24,24))
        l_Eye/=255
        l_Eye=l_Eye.reshape(24,24,-1)
        l_Eye = np.expand_dims(left_eye,axis=0)

        if(left_pred[0]==1):
                label='open'
        if(left_pred[0]==0):
                label = 'closed'
        break


    if(right_pred[0]==0 and left_pred[0]==0):
         reading+=1
         cv2.putText(frame,"closed",(10,height-20),font ,1,(255,155,233),1,cv2.LINE_AA)
        
    else:
        reading-=1
        cv2.putText(frame,'open',(10,height-20),font,1,(255,155,233),1,cv2.LINE_AA)

        

        # displaying the detection based on the conditions

    if(reading<0):
            reading=0
            cv2.putText(frame,"score" +str(reading),(100,height-20), font ,1,(255,123,234),1,cv2.LINE_AA)

    if(reading>20):
        cv2.imwrite(os.path.join(path,"detection.jpg"),frame)

        try:
                sound.play()

        except:
                pass
        if(thicc<16):
                thicc+=2
        else:
                thicc-=2
                if(thicc<2):
                    thicc=2

        cv2.rectangle(frame,(0,0),(width,height),(0,100,252),thicc)
    cv2.imshow('frame',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    cap.realse()
    cv2.destroyAllWindows()





        

