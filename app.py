
from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
import time

app = Flask(__name__)

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Data structures to hold trackers, labels, timers for each face, and talking status
trackers = []
labels = []
disappear_timers = []
talking_status = []
last_talking_time = []

def is_mouth_open(shape):
    entire_mouth_height = np.linalg.norm(shape[51] - shape[57])
    lip_distance = np.linalg.norm(shape[62] - shape[66])
    return lip_distance >= 0.20 * entire_mouth_height

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 24)
    
    speaker_count = 0  # To assign unique IDs to new faces
    disappear_threshold = 2.0  # Amount of time (in seconds) to wait before considering a face as gone
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update each tracker with the new frame
        trackers_to_remove = []
        for i, tracker in enumerate(trackers):
            quality = tracker.update(frame)
            if quality < 7:  # If tracker quality is low, update its disappear timer
                if time.time() - disappear_timers[i] > disappear_threshold:
                    trackers_to_remove.append(i)
            else:
                disappear_timers[i] = time.time()
        
        # Remove trackers that have been lost for a while and their associated data
        for i in sorted(trackers_to_remove, reverse=True):
            del trackers[i]
            del labels[i]
            del disappear_timers[i]
            del talking_status[i]
            del last_talking_time[i]
        
        # Detect faces if the number of trackers is less than the number of detected faces
        if len(trackers) < len(face_detector(gray)):
            faces = face_detector(gray)
            for rect in faces:
                if not any([tracker.get_position().contains(dlib.point(rect.center().x, rect.center().y)) for tracker in trackers]):
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, rect)
                    trackers.append(tracker)
                    speaker_count += 1
                    labels.append(f"Speaker {speaker_count}")
                    disappear_timers.append(time.time())
                    talking_status.append("Silent")
                    last_talking_time.append(None)

        # Draw rectangles and labels for each tracker and determine talking status
        speaking_speakers = []
        for i, tracker in enumerate(trackers):
            rect = tracker.get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, labels[i], (pt1[0], pt1[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            dlib_rect = dlib.rectangle(pt1[0], pt1[1], pt2[0], pt2[1])
            shape = predictor(gray, dlib_rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            for j, (x, y) in enumerate(shape):
                if j in [51, 57]:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                elif j in [62, 66]:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                else:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if is_mouth_open(shape):
                last_talking_time[i] = time.time()
                talking_status[i] = "Talking"
                speaking_speakers.append(labels[i])
                cv2.putText(frame, "Talking", (pt1[0], pt1[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif talking_status[i] == "Talking" and (last_talking_time[i] is None or time.time() - last_talking_time[i] < 1.5):
                talking_status[i] = "Talking"
                speaking_speakers.append(labels[i])
                cv2.putText(frame, "Talking", (pt1[0], pt1[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                talking_status[i] = "Silent"
                cv2.putText(frame, "Silent", (pt1[0], pt1[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display which speakers are talking
        if speaking_speakers:
            speaking_text = ", ".join(speaking_speakers) + " is/are speaking"
            cv2.putText(frame, speaking_text, (30, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
