import cv2
import os, sys
import numpy as np
import random
import faceRecognitionTraining as training
import json
from PIL import Image
    
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
    # print(who,accuracy)
    return (who,accuracy)


# def saveImgDataSet(path,number,face):
#     name = path+'\\subject'+str(number)+'.'+str(random.randrange(0,1000000))+'.png'
#     cv2.imwrite(name,face)
#     print(name)
    
def main(cam,isSaveDataSet=False):
    names = {}
    with open('datasets.json','r') as f:
        names = json.load(f)

    if isSaveDataSet:
        DATASET_NUMBER = names['lastnumber'] + 1
        DATASET_NAME = str(input("Name : "))
        DATASET_NEW_PATH = os.path.join(DATASET_PATH,str(DATASET_NUMBER))
        
        if not os.path.exists(DATASET_NEW_PATH):
            os.makedirs(DATASET_NEW_PATH)
    i=0
    while cam.isOpened():
        _,frame = cam.read()
        faces = faceDetection(frame)

        if faces != ():
            faces_img = getFace(frame,faces,True)
            found = []
            for face_img in faces_img:
                gray_faces_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
                if isSaveDataSet:
                    # cv2.imwrite(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.png',gray_faces_img)
                    Image.fromarray(gray_faces_img,mode='L').save(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.gif')
                    i+=1
                number, conf = recognize(gray_faces_img)
                found.append((names[str(number)],conf))
            print(found)
            frame = drawRec(frame,faces)

        cv2.imshow('Face',frame)
        if cv2.waitKey(1) == ord('q'):

            break

    if isSaveDataSet:
        names[str(DATASET_NUMBER)] = DATASET_NAME
        names['lastnumber'] = DATASET_NUMBER 
        json.dump(names,open('datasets.json','w+'),indent=4)
        print('collecting dataset DONE!')



if __name__ == '__main__':
    args = sys.argv[1:]

    isSaveDataSet = False
    if len(args) != 0:
        if args[0] in ['-c','collectDataSet','collectdataset']:
            isSaveDataSet = True

    cam = cv2.VideoCapture(0)

    main(cam,isSaveDataSet)
