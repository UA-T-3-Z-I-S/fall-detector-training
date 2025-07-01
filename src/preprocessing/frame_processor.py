import cv2
import numpy as np
import random

def resize_frame(frame, target_size=(224, 224)):
    return cv2.resize(frame, target_size)

def normalize_frame(frame):
    return frame.astype('float32') / 255.0

def augment_frame(frame):
    # Flip horizontal aleatorio
    if random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    # Cambio de brillo suave
    if random.random() < 0.3:
        factor = random.uniform(0.9, 1.1)  # brillo entre -10% y +10%
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)

    # Pequeña traslación
    if random.random() < 0.2:
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)

    # Ruido leve (gaussiano)
    if random.random() < 0.2:
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

    return frame

def preprocess_frame(frame, target_size=(224, 224), augment=False):
    frame = augment_frame(frame) if augment else frame
    resized = resize_frame(frame, target_size)
    normalized = normalize_frame(resized)
    return normalized
