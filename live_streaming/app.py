from flask import Flask, Response
import cv2, sys
# import logging as log
import datetime as dt
from time import sleep

app = Flask(__name__)

video = cv2.VideoCapture(0)

@app.route('/')
def index():
    return "Default Message"

def gen(video):
    
    while True:
        if not video.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        global anterior
        if anterior != len(faces):
            anterior = len(faces)
            # log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    import os
    os.chdir(os.getcwd())
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    global anterior
    anterior = 0
    # log.basicConfig(filename='webcam.log', level=log.INFO)
    app.run(host='0.0.0.0', port=2204, threaded=True)