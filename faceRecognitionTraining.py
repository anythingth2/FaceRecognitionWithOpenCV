import cv2
import numpy as np
import os
import main
from PIL import Image

recognizer =  cv2.face.LBPHFaceRecognizer_create()


def get_image_and_label(path):
    img_paths = [os.path.join(path,f) for f in os.listdir(path) if not f.endswith('.sad') or not f.endswith('.smile')]

    imgs = []
    labels = []
    for img_path in img_paths:
        # img = cv2.imread(img_path)
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

        
        nbr = int(os.path.split(img_path)[1].split(".")[0].replace("subject", ""))

        imgs.append(face)
        labels.append(nbr)
    return imgs,labels


if __name__ == '__main__':
    path = 'datasets\yalefaces'
    imgs,labels = get_image_and_label(path)

    recognizer.train(imgs,np.array(labels))
    recognizer.write('model.xml')


