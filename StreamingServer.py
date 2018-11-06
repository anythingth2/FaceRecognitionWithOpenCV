from flask import Flask, render_template,request, Response
import cv2
import RPi.GPIO as GPIO
from threading import Timer 
import time


cam = cv2.VideoCapture(0)
app = Flask(__name__)

doorPins = [2,3,4,17]
isOpening = False

GPIO.setmode(GPIO.BCM)
    
GPIO.setup(doorPins, GPIO.OUT,initial=GPIO.LOW)
#GPIO.cleanup()

def generateFrame():
    while cam.isOpened():
        _, frame = cam.read()
        print(type(frame))
        # yield frame
        _, buff = cv2.imencode('.jpg',frame)
        print(_,len(buff))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buff.tobytes() + b'\r\n')

@app.route('/index')
def index():
    return 'Hello'
@app.route('/vid')
def video_feed():
    return Response(generateFrame(),mimetype='multipart/x-mixed-replace; boundary=frame')
def openDoor(index):
    
    global isOpening
    if isOpening:
        return
    isOpening = True
    GPIO.output(doorPins[index],GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(doorPins[index],GPIO.LOW)
    time.sleep(5)
    print('open {}'.format((index)))
    isOpening = False
@app.route('/open',methods=['GET'])
def openDoorApi():
    doorIndex = request.args.get('door',type = int)
    openDoor(doorIndex)
    return Response()
@app.route('/image')
def image():
    ret, frame = cam.read()
    if(not ret):
        return 'fail to capture image'
    ret, buff = cv2.imencode('.jpg',frame)
    if(not ret):
        return 'fail to encode'
    return buff.tobytes()

if __name__ == '__main__':
    
    app.run(host='0.0.0.0',threaded=True)

        
