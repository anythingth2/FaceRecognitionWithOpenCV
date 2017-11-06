import cv2
import os
import numpy as np
import random

    
CLASSIFIER_PATH = os.getcwd()+'\\etc\\haarcascades\\haarcascade_frontalface_default.xml'
MODEL_PATH = os.getcwd() + '\model.xml'
DATASET_PATH = os.getcwd()+'\\datasets\\newfaces'



faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)
recognizer =  cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)


def drawRec(img,points):
    for point in points:
        x,y = (point[0],point[1])
        width,height = (point[2],point[3])
        cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),3)
    return img


def faceDetection(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return faceCascade.detectMultiScale(
        img,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize=(30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )


def getFace(img,faces = None,hasFace = False):
    if hasFace == False :
        faces = faceDetection(img)
    
    if faces == ():
        return []
    return [img[y:y+h,x:x+h] for (x,y,w,h) in faces ]

def recognize(face):
    who,accuracy = recognizer.predict(face)
    print(who,accuracy)
    return (who,accuracy)


def saveDataSet(path,number,face):
    name = path+'\\subject'+str(number)+'.'+str(random.randrange(0,1000000))+'.png'
    cv2.imwrite(name,face)
    print(name)
    


if __name__ == '__main__':
    
    cam = cv2.VideoCapture(0)
    isSaveDataSet = False
    if isSaveDataSet:
        DATESET_NUMBER = int(input('WHO ARE YOU : '))
        DATASET_NEW_PATH = os.path.join(DATASET_PATH,str(DATESET_NUMBER))
        if not os.path.exists(DATASET_NEW_PATH):
            os.makedirs(DATASET_NEW_PATH)

    while cam.isOpened():
        _,frame = cam.read()
        faces = faceDetection(frame)

        if faces != ():
            # print(faces)
            faces_img = getFace(frame,faces,True)
            
            # print(face)
            # cv2.imshow('face',face)
            for face_img in faces_img:
                gray_faces_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
                if isSaveDataSet:
                    saveDataSet(DATASET_NEW_PATH,DATESET_NUMBER,gray_faces_img)
                recognize(gray_faces_img)

            frame = drawRec(frame,faces)

        cv2.imshow('Face',frame)
        if cv2.waitKey(1) == ord('q'):
            break

