from flask import Flask, render_template, Response
import cv2
import numpy as np
import utlis

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 640)
cap.set(4, 480)

def generate_frames():
    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break

        imgContours, conts = utlis.getContours(frame, minArea=50000, filter=4)
        if len(conts) != 0:
            biggest = conts[0][2]
            imgWarp = utlis.warpImg(frame, biggest, 210, 297)
            imgContours2, conts2 = utlis.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)
            if len(conts2) != 0:
                for obj in conts2:
                    nPoints = utlis.reorder(obj[2])
                    nW = round((utlis.findDis(nPoints[0][0] // 3, nPoints[1][0] // 3) / 10), 1)
                    nH = round((utlis.findDis(nPoints[0][0] // 3, nPoints[2][0] // 3) / 10), 1)
                    x, y, w, h = obj[3]

                    # Annotate dimensions on the A4 sheet
                    cv2.putText(frame, '{}cm'.format(nW), (x + int(w/2), y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                    cv2.putText(frame, '{}cm'.format(nH), (x - 70, y + int(h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
