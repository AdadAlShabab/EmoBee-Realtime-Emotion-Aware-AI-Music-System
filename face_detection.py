import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 is the default webcam

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the image to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face detection
        results = face_detection.process(rgb_frame)

        # Draw face detection annotations
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Show the frame
        cv2.imshow('Real-Time Face Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
