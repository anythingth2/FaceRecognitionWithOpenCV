import cv2
import os, sys,shutil
import time
import numpy as np
import random
import faceRecognitionTraining as training
import json
from PIL import Image
# from io import StringIO
# import io
import requests
from threading import Thread
CLASSIFIER_PATH = os.getcwd()+'\\etc\\haarcascades\\haarcascade_frontalface_default.xml'
MODEL_PATH = os.getcwd() + '\model.xml'
DATASET_PATH = os.getcwd()+'\\datasets\\newfaces'

SERVER_URL = 'http://192.168.1.26:5000'
SERVER_DOOR_URL = 'http://192.168.0.26:5001'
VIDEOSTREAM_PATH = '/vid'
OPENDOOR_PATH = '/open'

faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)


recognizer =  cv2.face.LBPHFaceRecognizer_create()

def tic():
    global t
    t = time.time()

def toc():
    global t
    print('time elapsed',(time.time()-t)*1000,'milliseconds')
    t = time.time()

def drawRec(img,point,text=None,color=None):
    # for point in points:
    x,y = (point[0],point[1])
    width,height = (point[2],point[3])
    if color == None:
        color = (0,255,0)
    cv2.rectangle(img,(x,y),(x+width,y+height),color,2)
    if text != None:
        cv2.putText(img,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)
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
    x,y,w,h = faces
    return img[y:y+h,x:x+h]

def recognize(face):
    ID,conf = recognizer.predict(face)
    # print(who,accuracy)
    return (ID,conf)

def loadDatasetID():
    with open('datasets.json','r') as f:
        return json.load(f)

def clearModel():
    print('start clearing model...')

    open(MODEL_PATH,'w').close()

    dataset_json = loadDatasetID()
    lastestID = dataset_json['lastnumber']
    json.dump({'lastnumber':lastestID},open('datasets.json','w'))

    for folder in os.listdir(DATASET_PATH):shutil.rmtree(os.path.join(DATASET_PATH,folder))

    print('CLEARED!')

def loadModel(recogniz):
    print('loading recognition model...')
    try:
        recogniz.read(MODEL_PATH)
        # recogniz.setThreshold(60)
    except cv2.error:
        print('model not found!')
        exit()
    print('Starting!')


def openDoor(doorId):
    doorReq = requests.get(SERVER_DOOR_URL+OPENDOOR_PATH,params = {'door':doorId})
# def saveImgDataSet(path,number,face):
#     name = path+'\\subject'+str(number)+'.'+str(random.randrange(0,1000000))+'.png'
#     cv2.imwrite(name,face)
#     print(name)
    
def main(cam,isSaveDataSet=False):
    if not cam.isOpened():
        print('Camera was not open')
        exit()
    names = loadDatasetID()
    if isSaveDataSet:
        DATASET_NUMBER = names['lastnumber'] + 1
        DATASET_NAME = str(input("Name : "))
        DATASET_DOORID = int(input("DOOR ID (1-4): "))
        DATASET_NEW_PATH = os.path.join(DATASET_PATH,str(DATASET_NUMBER))
        
        if not os.path.exists(DATASET_NEW_PATH):
            os.makedirs(DATASET_NEW_PATH)
    i=0

    while cam.isOpened():
        _,frame = cam.read()

        faces = faceDetection(frame)
        
        if faces != ():
            face = faces[0]
            face_img = getFace(frame,face,True)
            # found = []
            # tic()
            # for face_img in faces_img:
            #     gray_faces_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
            #     if isSaveDataSet:
            #         # cv2.imwrite(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.png',gray_faces_img)
            #         Image.fromarray(gray_faces_img,mode='L').save(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.gif')
            #         i+=1
            #     else:
            #         number, conf = recognize(gray_faces_img)
            #         found.append((names[str(number)],conf))
            gray_face_img = cv2.cvtColor(face_img ,cv2.COLOR_BGR2GRAY)
            if isSaveDataSet:
                frame = drawRec(frame,face)
                cv2.imwrite(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.png',gray_face_img)
                i+=1
            else:
                ID,conf = recognize(gray_face_img)
                # print(names[str(ID),conf])
                print(names[str(ID)],conf)
                if conf < 90:
                    frame = drawRec(frame,face,names[str(ID)]['name']+' '+str(conf))
                else:
                    frame = drawRec(frame,face,color=(0,0,255))
                
        cv2.imshow('ESL - Embedded System Laboratory',frame)
        if cv2.waitKey(1) == ord('q'):
            break

    if isSaveDataSet:
        names[str(DATASET_NUMBER)] = {'name':DATASET_NAME,'door':DATASET_DOORID}
        names['lastnumber'] = DATASET_NUMBER 
        json.dump(names,open('datasets.json','w+'),indent=4)
        print('collecting dataset DONE!')

def mainWithSocket(isSaveDataSet = False):
    names = loadDatasetID()
    if isSaveDataSet:
        DATASET_NUMBER = names['lastnumber'] + 1
        DATASET_NAME = str(input("Name : "))
        DATASET_DOORID = int(input("DOOR ID (1-4): "))
        DATASET_NEW_PATH = os.path.join(DATASET_PATH,str(DATASET_NUMBER))
        
        if not os.path.exists(DATASET_NEW_PATH):
            os.makedirs(DATASET_NEW_PATH)

    count = 0
    # reqDoor = requests.post(SERVER_URL+OPENDOOR_PATH)
    print('connecting...')
    r = requests.get(SERVER_URL+VIDEOSTREAM_PATH,stream=True)
    print('connected')
    i=0
    byte=b''
    while True:
        
        cancel = False
        frame = np.zeros((2,2),dtype='uint8')
        while not cancel:
            try:
                byte += r.raw.read(1024)
                a = byte.find(b'\xff\xd8')
                b = byte.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = byte[a:b+2]
                    byte = byte[b+2:]
                    # img = cv2.imdecode(np.fromstring(jpg,dtype=np.uint8,),cv2.IMREAD_COLOR)
                    frame = cv2.imdecode(np.fromstring(jpg,dtype=np.uint8,),cv2.IMREAD_COLOR)
                    # frame = np.array(frame,dtype='uint8')
                    break
            except:
                print('Failed to get image from server')
                cancel = True

        # frame = np.load(io.BytesIO(ultimate_buffer))['frame']

        faces = faceDetection(frame)
        # faces = faceDetection(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        if faces != ():
            face = faces[0]
            face_img = getFace(frame,face,True)
            # found = []
            # tic()
            # for face_img in faces_img:
            #     gray_faces_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
            #     if isSaveDataSet:
            #         # cv2.imwrite(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.png',gray_faces_img)
            #         Image.fromarray(gray_faces_img,mode='L').save(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.gif')
            #         i+=1
            #     else:
            #         number, conf = recognize(gray_faces_img)
            #         found.append((names[str(number)],conf))
            gray_face_img = cv2.cvtColor(face_img ,cv2.COLOR_BGR2GRAY)
            if isSaveDataSet:
                
                frame = drawRec(frame,face)
                cv2.imwrite(DATASET_NEW_PATH+'\\subject'+str(DATASET_NUMBER)+'.'+str(i)+'.png',gray_face_img)
                i+=1
            else:
                pass
                ID,conf = recognize(gray_face_img)
                print(names[str(ID)],conf)
                if conf < 60:
                    frame = drawRec(frame,face,names[str(ID)]['name']+' '+str(conf))
                    print('open ',names[str(ID)]['door'])
                    openDoorThread = Thread(target= openDoor,args=(names[str(ID)]['door'],))
                    openDoorThread.daemon = True
                    openDoorThread.start()
                else:
                    frame = drawRec(frame,face,color=(0,0,255))
                
        cv2.imshow('ESL - Embedded System Laboratory',frame)
        if cv2.waitKey(1) == ord('q'):
            break

    if isSaveDataSet:
        names[str(DATASET_NUMBER)] = {'name':DATASET_NAME,'door':DATASET_DOORID}
        names['lastnumber'] = DATASET_NUMBER 
        json.dump(names,open('datasets.json','w+'),indent=4)
        print('collecting dataset DONE!')
    

def quickstart():
    # print('camera starting...')
    # cam = cv2.VideoCapture(0)
    # if cam.isOpened():
    #     print('camera opened')
    # else:
    #     print('camera is not starting')
    #     exit()

    print('Collect Dataset')
    print('Stop when press q in VideoScreen')
    while True:
        # main(cam,isSaveDataSet=True)
        mainWithSocket(isSaveDataSet=True)
        if not input('Do you want collect dataset again? (yes,no) : ') in ['yes','y']:
            break
    print('Start training dataset')
    from faceRecognitionTraining import train
    train()
    print('DONE!!!'.center(20,'-'))    


if __name__ == '__main__':
    args = sys.argv[1:]

    isSaveDataSet = False
    if len(args) != 0:
        if args[0] in ['-c','collectDataSet','collectdataset']:
            # loadModel(recognizer)
            isSaveDataSet = True
            cam = cv2.VideoCapture(0)
            main(cam,isSaveDataSet)
        elif args[0] in ['--cm','clearModel']:
            clearModel()
        elif args[0] in ['-t','train','trainModel']:
            from faceRecognitionTraining import train
            train()
        elif args[0] in ['-q','quick','quickstart']:
            quickstart()

    else:
        loadModel(recognizer)
        # cam = cv2.VideoCapture(0)
        # main(cam)
        mainWithSocket()