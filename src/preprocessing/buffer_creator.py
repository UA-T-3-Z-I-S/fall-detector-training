import numpy as np
from src.preprocessing.frame_processor import preprocess_frame

def create_buffers(frames, buffer_size=16, overlap=0.3, target_size=(224, 224), augment=False):
    step = int(buffer_size * (1 - overlap))
    buffers = []

    for index in range(0, len(frames) - buffer_size + 1, step):
        buffer = frames[index:index + buffer_size]
        processed_buffer = [preprocess_frame(f, target_size, augment=augment) for f in buffer]
        buffers.append(np.array(processed_buffer))

    return buffers
