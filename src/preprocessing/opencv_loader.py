import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.config.paths import DATASET_CAIDA, DATASET_NO_CAIDA, BUFFER_CAIDA, BUFFER_NO_CAIDA
from src.preprocessing.frame_processor import preprocess_frame

def process_videos(input_path, output_path, buffer_size=16, overlap=0.3, target_size=(224, 224)):
    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi'))]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for video_file in video_files:
        process_single_video(video_file, input_path, output_path, buffer_size, overlap, target_size)

def process_single_video(video_file, input_path, output_path, buffer_size=16, overlap=0.3, target_size=(224, 224)):
    video_path = os.path.join(input_path, video_file)
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    step = int(buffer_size * (1 - overlap))
    index = 0
    buffer_count = 0

    while index + buffer_size <= len(frames):
        buffer = frames[index:index + buffer_size]

        processed_buffer = []
        for frame in buffer:
            processed_frame = preprocess_frame(frame, target_size)
            processed_buffer.append(processed_frame)

        processed_buffer = np.array(processed_buffer)

        save_path = os.path.join(output_path, f'{os.path.splitext(video_file)[0]}_buffer_{buffer_count}.npy')
        np.save(save_path, processed_buffer)

        buffer_count += 1
        index += step

    print(f'Procesado: {video_file} - {buffer_count} buffers generados')

def run_processing():
    print('Procesando videos de caídas...')
    video_files_caida = [f for f in os.listdir(DATASET_CAIDA) if f.endswith(('.mp4', '.avi'))]

    print('Procesando videos sin caídas...')
    video_files_no_caida = [f for f in os.listdir(DATASET_NO_CAIDA) if f.endswith(('.mp4', '.avi'))]

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Procesar caídas
        for video_file in video_files_caida:
            executor.submit(process_single_video, video_file, DATASET_CAIDA, BUFFER_CAIDA)

        # Procesar no caídas
        for video_file in video_files_no_caida:
            executor.submit(process_single_video, video_file, DATASET_NO_CAIDA, BUFFER_NO_CAIDA)

    print('Procesamiento finalizado.')
