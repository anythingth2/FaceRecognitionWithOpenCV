import cv2
import numpy as np
import os
import main
from PIL import Image

recognizer =  cv2.face.LBPHFaceRecognizer_create()


def get_image_and_label(path):
    img_folders = [os.path.join(path,f) for f in os.listdir(path) ]

    imgs = []
    labels = []
    for img_folder in img_folders:
        # img = cv2.imread(img_path)
        
        number = int(os.path.split(img_folder)[-1])
        img_paths = [os.path.join(img_folder,f) for f in os.listdir(img_folder)]
        for img_path in img_paths:

            img = Image.open(img_path)
            img = np.array(img,'uint8')
        # cv2.imshow('fa',img)
        # cv2.waitKey(0)
            if img.size == 0:breaks
            
            face = main.getFace(img)
            if len(face) == 1:
                face = main.getFace(img)[0]
            else:continue
            
            if len(face.shape) == 3:
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

            imgs.append(face)
            labels.append(number)
            print(number , img_path)
    return imgs,labels

def train():
    path = 'datasets\\newfaces'
    imgs,labels = get_image_and_label(path)

    recognizer.train(imgs,np.array(labels))
    recognizer.write('model.xml')
if __name__ == '__main__':
    train()


