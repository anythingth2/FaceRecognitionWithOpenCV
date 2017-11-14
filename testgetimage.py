import requests
import cv2
import numpy as np
def run(r):
    byte=b''
    cancel = False
    while not cancel:
        try:
            print('try reading')
            byte += r.raw.read(1024)
            print('readed')
            a = byte.find(b'\xff\xd8')
            b = byte.find(b'\xff\xd9')
            if a != -1 and b != -1:
                print('found img')
                jpg = byte[a:b+2]
                byte = byte[b+2:]
                img = cv2.imdecode(np.fromstring(jpg,dtype=np.uint8,),cv2.IMREAD_COLOR)
                cv2.imshow('frame',img)
                if cv2.waitKey(1) == ord('q'):cancel = True
        except IOError:
            print('exception!')
            cancel = True
r = requests.get('http://192.168.0.138:5000/videoStream/',stream=True)
run(r)