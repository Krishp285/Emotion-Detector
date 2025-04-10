# Real-Time Emotion Detection by [Your Name]
import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model("C:/emotion_detector_project/emotion_model.h5")

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Haar Cascade not loaded!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam error!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        prediction = model.predict(face)[0]
        emotion = emotions[np.argmax(prediction)]
        confidence = prediction[np.argmax(prediction)]

        label = f"{emotion} ({confidence:.2f})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()