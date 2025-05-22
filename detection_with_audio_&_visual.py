import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import pygame

# Load emotion model
model = load_model('emotion_model.h5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize Pygame for sound feedback
pygame.mixer.init()
ding_sound = pygame.mixer.Sound("music.wav")

# Timing and state variables
last_prediction_time = 0
prediction_interval = 10  # seconds
last_detected_emotion = None
last_confidence = 0.0
flash_effect = False
flash_duration = 10  # frames
flash_counter = 0

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Draw scanning dots
        for i, landmark in enumerate(face_landmarks.landmark):
            if i % 5 == 0:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (144, 238, 144), -1)

        # Bounding box for face crop
        x_coords = [int(lm.x * frame.shape[1]) for lm in face_landmarks.landmark]
        y_coords = [int(lm.y * frame.shape[0]) for lm in face_landmarks.landmark]
        x_min, x_max = max(min(x_coords), 0), min(max(x_coords), frame.shape[1])
        y_min, y_max = max(min(y_coords), 0), min(max(y_coords), frame.shape[0])

        face = frame[y_min:y_max, x_min:x_max]
        if face.size != 0:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (64, 64))
            face_input = face_resized / 255.0
            face_input = np.expand_dims(face_input, axis=(0, -1))

            current_time = time.time()
            if current_time - last_prediction_time > prediction_interval:
                last_prediction_time = current_time

                predictions = model.predict(face_input)
                emotion = emotion_labels[np.argmax(predictions)]
                confidence = np.max(predictions)

                last_detected_emotion = emotion
                last_confidence = confidence

                # Feedback
                ding_sound.play()
                flash_effect = True
                flash_counter = flash_duration

    # Flash frame border when detection happens
    if flash_effect:
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        flash_counter -= 1
        if flash_counter <= 0:
            flash_effect = False

    # Display detected emotion
    if last_detected_emotion:
        text = f"{last_detected_emotion} ({last_confidence * 100:.1f}%)"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x = frame.shape[1] - text_size[0] - 20
        y = 50
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    # Countdown message
    remaining = int(max(0, prediction_interval - (time.time() - last_prediction_time)))
    cv2.putText(frame, f"Hold emotion... {remaining}s",
                (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)

    # Display frame
    cv2.imshow("Emotion Aware System", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
