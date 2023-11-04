from fastapi import FastAPI, Response
from threading import Thread
import cv2
import os
from typing import Optional

app = FastAPI()

def drowsiness_detection():
    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

    lbl=['Close','Open']
    model = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y+h, x:x+w]
            count += 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = r_eye.reshape(24, 24, -1)
            rpred = model.predict_classes(r_eye)
            if rpred[0] == 1:
                lbl='Open' 
            if rpred[0] == 0:
                lbl='Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y+h, x:x+w]
            count += 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            lpred = model.predict_classes(l_eye)
            if lpred[0] == 1:
                lbl='Open'   
            if lpred[0] == 0:
                lbl='Closed'
            break

        if rpred[0] == 0 and lpred[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0

        if score > 15:
            cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
            break

    cap.release()
    cv2.destroyAllWindows()

@app.get("/")
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Drowsiness Detection System</h1>
        <button onclick="startDetection()">Start Detection</button>

        <script>
        function startDetection() {
            fetch('/start_detection')
            .then(response => response.text())
            .then(data => alert(data));
        }
        </script>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

@app.get("/start_detection")
def start_detection():
    t = Thread(target=drowsiness_detection)
    t.start()
    return "Drowsiness detection started."