import cv2
import numpy as np
import face_recognition



jinal=face_recognition.load_image_file('ImagesBasic/Jinal Shah.jpg')
jinal=cv2.cvtColor(jinal,cv2.COLOR_BGR2RGB)
jinaltest1=face_recognition.load_image_file('ImagesBasic/Dhruvi Shah.jpeg')
jinaltest1=cv2.cvtColor(jinaltest1,cv2.COLOR_BGR2RGB)


faceloc=face_recognition.face_locations(jinal)[0]
encodejinal=face_recognition.face_encodings(jinal)[0]
cv2.rectangle(jinal,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloc=face_recognition.face_locations(jinaltest1)[0]
encodeTest=face_recognition.face_encodings(jinaltest1)[0]
cv2.rectangle(jinaltest1,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodejinal],encodeTest)
faceDis=face_recognition.face_distance([encodejinal],encodeTest)
print(results,faceDis)

cv2.putText(jinaltest1,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

imgS = cv2.resize(jinal,(0,0),None,1.0,1.0)
cv2.imshow('jinal Shah',imgS)
# cv2.imshow('jinal Shah',jinal).resize
# imgS = cv2.resize(img, (0, 0), None, 1.0, 1.0)
imgS1 = cv2.resize(jinaltest1,(0,0),None,1.0,1.0)
cv2.imshow('jinal test',imgS1)

cv2.waitKey(0)
cv2.destroyAllWindows()