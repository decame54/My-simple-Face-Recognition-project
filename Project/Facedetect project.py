import cv2
from face_recognition.api import face_distance
import numpy as np
import face_recognition
import os
from datetime import datetime

path ='Picture'
Picture = []
Name = []
List = os.listdir(path)
print(List)
for classname in List:
    CurrentImage = cv2.imread(f'{path}/{classname}')
    Picture.append(CurrentImage)
    Name.append(os.path.splitext(classname)[0])
print(Name)

def findEncoding(Picture):
    encodeList = []
    for img in Picture:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeimg)
    return encodeList

def mark(matchname):
    with open('Attendance.csv','r+') as f:
        mydataList = f.readlines()
        nameList = []
        #print(mydataList)
        for line in mydataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if matchname not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{matchname},{ dtString}')

encodeListKnown = findEncoding(Picture)
print('Encoding Complete')

cap = cv2.VideoCapture(1)
while True:
    success, img = cap.read()
    imgcap = cv2.resize(img,(0,0),None,0.25,0.25)
    imgcap = cv2.cvtColor(imgcap, cv2.COLOR_BGR2RGB)
    FaceCurrentframe = face_recognition.face_locations(imgcap)
    encodeCurrentframe = face_recognition.face_encodings(imgcap,FaceCurrentframe)

    for encodeFace, faceLocation in zip(encodeCurrentframe, FaceCurrentframe):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDistance)
        #print(faceDistance)
        matchindex = np.argmin(faceDistance)

        if faceDistance[matchindex] < 0.40:
            matchname = Name[matchindex].upper()
            #print(matchname)
            mark(matchname) 
        else: matchname = 'unknown'
        y1,x2,y2,x1 = faceLocation
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.rectangle(img,(x1,y2 - 35),(x2,y2),(255,0,0),cv2.FILLED)
        cv2.putText(img, matchname,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


