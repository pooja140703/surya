from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load model and labels
model = tf.keras.models.load_model("cnn_pose_classifier.h5")
class_labels = [
    "adho mukh svanasana",
    "ashtanga namaskara",
    "ashwa sanchalanasana",
    "bhujangasana",
    "hasta utthanasana",
    "kumbhakasana",
    "padahastasana",
    "pranamasana"
]

# Load MoveNet
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet.signatures['serving_default']

POSE_CONNECTIONS = [
    (0, 1), (1, 3), (0, 2), (2, 4), 
    (5, 7), (7, 9), (6, 8), (8, 10), 
    (5, 6), (5, 11), (6, 12), 
    (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 12)
]

def draw_landmarks(img, keypoints, threshold=0.3):
    h, w, _ = img.shape
    for i, (y, x, c) in enumerate(keypoints):
        if c > threshold:
            cv2.circle(img, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)
    for (i, j) in POSE_CONNECTIONS:
        y1, x1, c1 = keypoints[i]
        y2, x2, c2 = keypoints[j]
        if c1 > threshold and c2 > threshold:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)

def gen_frames():
    cap = cv2.VideoCapture(0)
    conf_threshold = 0.8
    kp_threshold = 0.3
    min_valid_kp = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = tf.image.resize_with_pad(tf.convert_to_tensor(img_rgb), 192, 192)
        input_tensor = tf.expand_dims(tf.cast(img_input, dtype=tf.int32), axis=0)

        outputs = movenet(input_tensor)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        valid_kp = np.sum(keypoints[:, 2] > kp_threshold)

        if valid_kp >= min_valid_kp:
            input_data = keypoints.flatten().reshape(1, 17, 3, 1)
            prediction = model.predict(input_data)
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > conf_threshold:
                label = class_labels[pred_class]
            else:
                label = ""
        else:
            label = ""

        draw_landmarks(frame, keypoints)
        cv2.putText(frame, f"{label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0) if label != "No Pose Detected" else (0, 0, 255), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
