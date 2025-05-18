# Realtime-Emotion-Aware-Music-System
┌──────────┐    ┌───────────────┐    ┌────────────┐
│ Webcam   │ -> │ Preprocessor  │ -> │ Face Model │
│ (OpenCV) │    │ (Crop, Resize)│    │ (MediaPipe)│
└──────────┘    └───────────────┘    └────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │Emotion Model │
                                    │ (Classifier) │
                                    └──────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ Playlist Mapper & API   │
                              │ (Spotify auth, choose   │
                              │  playlist, control)     │
                              └─────────────────────────┘
