import cv2

def camera():
    cam = cv2.VideoCapture(0)
    
    while cam.isOpened():
        _,frame = cam.read()
        