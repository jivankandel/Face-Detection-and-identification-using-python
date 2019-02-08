import numpy as np 
import cv2
import pickle
cap =cv2.VideoCapture(0)
recognizer =cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\Omen\Desktop\\AI\Project\\trainer.yml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\Omen\\Desktop\\AI\\Project\\haarcascade_frontalface_alt2.xml')
labels={"person_name":1}
with open("C:\\Users\\Omen\\Desktop\\AI\\Project\\labels.pickle",'rb') as f:
    labels=pickle.load(f)
    labels = {v:k for k,v in labels.items()}
    print(labels)
while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray)
    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h ,x:x+w]
        roi_color=frame[y:y+h ,x:x+w]
        id_, conf=recognizer.predict(roi_gray)
        print(id_)
        if  conf<=75:
            font= cv2.FONT_HERSHEY_COMPLEX
            name=labels[id_]+"  "+ str(conf)
            color=(255,255,255)
            stroke =2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        color= (255,0,0)
        stroke = 2
        width=x + w
        height=y + h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
    cv2.imshow('Recognizer',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
