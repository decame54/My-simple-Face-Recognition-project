import cv2
import face_recognition.api as face_recognition 
import numpy as np

Object = face_recognition.load_image_file('Picture/Manh.jpg')
Object = cv2.cvtColor(Object, cv2.COLOR_BGR2RGB)
Test = face_recognition.load_image_file('Test/ManhTest.jpg')
Test = cv2.cvtColor(Test, cv2.COLOR_BGR2RGB)

FaceLocation = face_recognition.face_locations(Object)
encodes = face_recognition.face_encodings(Object)
print(encodes)
print(FaceLocation)
for (top, right, bottom, left) in FaceLocation:
    cv2.rectangle(Object,(left,top),(right,bottom),(255,0,0),3)

FaceLocationTest = face_recognition.face_locations(Test)[0]
encodeTest = face_recognition.face_encodings(Test)[0]
cv2.rectangle(Test,(FaceLocationTest[3],FaceLocationTest[0]),(FaceLocationTest[1],FaceLocationTest[2]),(0,255,0),2) 

for encode in encodes:
    Result = face_recognition.compare_faces([encode],encodeTest)
    FaceDistance = face_recognition.face_distance([encode], encodeTest)
    print(Result, FaceDistance)
    cv2.putText(Test, f'{Result}{round(FaceDistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv2.imshow('Object', Object)
cv2.imshow('Test', Test)
cv2.waitKey(0)