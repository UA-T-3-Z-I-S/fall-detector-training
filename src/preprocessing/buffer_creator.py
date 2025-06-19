import numpy as np
from src.preprocessing.frame_processor import preprocess_frame

def create_buffers(frames, buffer_size=16, overlap=0.3, target_size=(224, 224)):
    step = int(buffer_size * (1 - overlap))
    index = 0
    buffers = []
    buffer_count = 0

    while index + buffer_size <= len(frames):
        buffer = frames[index:index + buffer_size]

        processed_buffer = []
        for frame in buffer:
            processed_frame = preprocess_frame(frame, target_size)
            processed_buffer.append(processed_frame)

        processed_buffer = np.array(processed_buffer)
        buffers.append(processed_buffer)

        buffer_count += 1
        index += step

    return buffers
