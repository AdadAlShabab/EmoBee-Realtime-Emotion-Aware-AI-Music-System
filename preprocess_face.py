import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture(0)

# Model input dimensions (you can change this based on your model)
IMG_SIZE = 48  # e.g., FER2013 uses 48x48 grayscale

def preprocess_face(frame, bbox):
    h, w, _ = frame.shape
    # Convert relative coordinates to absolute pixel values
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    w_box = int(bbox.width * w)
    h_box = int(bbox.height * h)
    
    # Add padding if needed
    margin = 10
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    x2 = min(x + w_box + 2*margin, frame.shape[1])
    y2 = min(y + h_box + 2*margin, frame.shape[0])

    # Crop face ROI
    face_roi = frame[y:y2, x:x2]

    # Resize to model input size
    face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))

    # Convert to grayscale (if required)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to [0, 1]
    face_normalized = face_gray / 255.0

    # Expand dimensions for model (1, 48, 48, 1)
    face_input = np.expand_dims(face_normalized, axis=(0, -1)).astype(np.float32)

    return face_input, face_resized  # For model and visualization

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Preprocess the face
                face_input, preview = preprocess_face(frame, bbox)

                # Show the preprocessed face (resized, grayscale)
                cv2.imshow("Preprocessed Face (48x48)", preview)

                # Optional: Draw bounding box on main feed
                mp_drawing.draw_detection(frame, detection)

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
