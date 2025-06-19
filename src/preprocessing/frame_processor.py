import cv2

def resize_frame(frame, target_size=(224, 224)):
    return cv2.resize(frame, target_size)

def normalize_frame(frame):
    return frame / 255.0

def preprocess_frame(frame, target_size=(224, 224)):
    resized = resize_frame(frame, target_size)
    normalized = normalize_frame(resized)
    return normalized
