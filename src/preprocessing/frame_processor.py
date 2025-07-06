import cv2
import numpy as np
import random

def resize_frame(frame, target_size=(224, 224)):
    return cv2.resize(frame, target_size)

def normalize_frame(frame):
    return frame.astype('float32') / 255.0

def augment_frame(frame):
    # Flip horizontal aleatorio
    if random.random() < 0.3:  # antes 0.5
        frame = cv2.flip(frame, 1)

    # Cambio de brillo suave
    if random.random() < 0.15:  # antes 0.3
        factor = random.uniform(0.9, 1.1)
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)

    # Pequeña traslación
    if random.random() < 0.1:  # antes 0.2
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)

    # Ruido leve (gaussiano)
    if random.random() < 0.1:  # antes 0.2
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

    # Rotación aleatoria
    if random.random() < 0.1:  # antes 0.2
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), angle, 1)
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)

    # Zoom aleatorio
    if random.random() < 0.1:  # antes 0.2
        scale = random.uniform(0.9, 1.1)
        h, w = frame.shape[:2]
        nh, nw = int(h*scale), int(w*scale)
        frame = cv2.resize(frame, (nw, nh))
        if scale < 1:
            pad_h = (h - nh) // 2
            pad_w = (w - nw) // 2
            frame = cv2.copyMakeBorder(frame, pad_h, h-nh-pad_h, pad_w, w-nw-pad_w, cv2.BORDER_REFLECT)
        else:
            frame = frame[:h, :w]

    return frame

def preprocess_frame(frame, target_size=(224, 224), augment=False):
    frame = augment_frame(frame) if augment else frame
    resized = resize_frame(frame, target_size)
    normalized = normalize_frame(resized)
    return normalized
