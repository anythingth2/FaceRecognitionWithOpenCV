import cv2
import numpy as np
import os
import main
from PIL import Image
import time

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
            if img.size == 0:break
            
            # face = main.getFace(img[0])
            # if len(face) == 1:
            #     face = main.getFace(img)
            # else:continue
            # if face == []:
                # continue
            if len(img.shape) == 3:
                face = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            else:
                face = img

            imgs.append(face)
            labels.append(number)
            print(number , img_path)
    return imgs,labels

def train():
    path = 'datasets\\newfaces'
    imgs,labels = get_image_and_label(path)
    print('start training model')
    time.sleep(0.5)
    recognizer.train(imgs,np.array(labels))
    print('training model end')
    print('saving model....')
    recognizer.write('model.xml')
    print('saving model DONE!')
if __name__ == '__main__':
    train()


