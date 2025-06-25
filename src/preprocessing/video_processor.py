import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.preprocessing.loader_opencv import load_video_frames
from src.preprocessing.buffer_creator import create_buffers

def process_videos(input_path, output_path, label, buffer_size=16, overlap=0.3, target_size=(224, 224)):
    print(f"📦 Procesando videos de '{label}' en {input_path}...")

    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi'))]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for video_file in video_files:
            executor.submit(
                process_single_video,
                video_file,
                input_path,
                output_path,
                buffer_size,
                overlap,
                target_size,
                label
            )

def process_single_video(video_file, input_path, output_path, buffer_size=16, overlap=0.3, target_size=(224, 224), label=None):
    video_path = os.path.join(input_path, video_file)

    try:
        frames = load_video_frames(video_path)

        buffers = create_buffers(frames, buffer_size, overlap, target_size)

        for idx, buffer in enumerate(buffers):
            save_path = os.path.join(output_path, f'{os.path.splitext(video_file)[0]}_buffer_{idx}.npy')
            np.save(save_path, buffer)

        print(f'✅ Procesado: {video_file} → {len(buffers)} buffers generados')

    except Exception as e:
        print(f'❌ Error procesando {video_file}: {e}')
