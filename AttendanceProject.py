import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendence(name):
    with open('AttendanceProject.csv', 'r+')as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findencodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:

    try:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 1.0, 1.0)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendence(name)

            cv2.imshow('Webcam', img)
            cv2.waitKey(1)


    except Exception as e:
        print(str(e))



# faceloc=face_recognition.face_locations(saumya)[0]
# encodeSaumya=face_recognition.face_encodings(saumya)[0]
# cv2.rectangle(saumya,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
#
# faceloc=face_recognition.face_locations(saumyatest1)[0]
# encodeTest=face_recognition.face_encodings(saumyatest1)[0]
# cv2.rectangle(saumyatest1,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
#
# results=face_recognition.compare_faces([encodeSaumya],encodeTest)
# faceDis=face_recognition.face_distance([encodeSaumya],encodeTest)
