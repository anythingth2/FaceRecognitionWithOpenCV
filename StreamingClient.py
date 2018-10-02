import requests
import numpy as np
import cv2

SERVER_URL = 'http://localhost:5000'+'/vid'
print('connecting')
r = requests.get(SERVER_URL,stream=True)
print('connected')
byte = b''
while True:
    cancel = False
    frame = np.zeros((2,2),dtype='uint8')
    while not cancel:
        try:
            incomingByte = r.raw.read(1024)
            byte += incomingByte
            a = byte.find(b'\xff\xd8')
            b = byte.find(b'\xff\xd9')
            # print('get buffer')
            if a != -1 and b != -1:
                jpg = byte[a:b+2]
                byte = byte[b+2:]
                frame = cv2.imdecode(np.fromstring(jpg,dtype=np.uint8,),cv2.IMREAD_COLOR)
                break
        except:
            print('Failed to get image from server')
            cancel = True
    cv2.imshow('f',frame)
    cv2.waitKey(1)
