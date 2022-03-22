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

input = cv2.VideoCapture("Chủ's Cursed Images.mp4")
length = int(input.get(cv2.CAP_PROP_FRAME_COUNT))
process = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('Output.mp4', process, 29.97, (1280, 720))

encodeListKnown = findEncoding(Picture)
print('Encoding Complete')
print('Input video complete')

frame_number = 0
while True:
    ret, frame = input.read()
    frame_number += 1

    if not ret:
        break
    rgb_frame = cv2.resize(frame,(0,0),None,0.25,0.25)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    FaceCurrentframe = face_recognition.face_locations(rgb_frame)
    encodeCurrentframe = face_recognition.face_encodings(rgb_frame, FaceCurrentframe)
    for encodeFace, faceLocation in zip(encodeCurrentframe, FaceCurrentframe):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDistance)
        #print(faceDistance)
        matchindex = np.argmin(faceDistance)

        if faceDistance[matchindex] < 0.50:
            matchname = Name[matchindex].upper()
        else: matchname ='Unknown'  
        y1,x2,y2,x1 = faceLocation
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.rectangle(frame,(x1,y2 - 35),(x2,y2),(255,0,0),cv2.FILLED)
        cv2.putText(frame, matchname,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    print("Writing frame {} / {}".format(frame_number, length))
    output.write(frame)    

input.release()
cv2.destroyAllWindows()
