import cv2

def preprocess_frame(frame, target_size=(224, 224)):
    # Redimensiona el frame
    resized_frame = cv2.resize(frame, target_size)
    # Normaliza el frame (valores entre 0 y 1)
    normalized_frame = resized_frame / 255.0
    return normalized_frame
