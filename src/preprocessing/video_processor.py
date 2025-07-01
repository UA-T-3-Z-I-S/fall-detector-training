import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.preprocessing.loader_opencv import load_video_frames
from src.preprocessing.buffer_creator import create_buffers
import subprocess
import tempfile
import shutil

def convertir_avi_a_mp4_temporal(ruta_avi):
    temp_dir = tempfile.mkdtemp()
    ruta_mp4 = os.path.join(temp_dir, os.path.splitext(os.path.basename(ruta_avi))[0] + ".mp4")
    comando = [
        'ffmpeg', '-y', '-i', ruta_avi,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'aac', '-b:a', '128k',
        ruta_mp4
    ]
    try:
        subprocess.run(comando, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ruta_mp4, temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Error convirtiendo {ruta_avi} a mp4: {e}")

def process_videos(input_path, output_path, label, buffer_size=16, overlap=0.3, target_size=(224, 224), augment=False):
    print(f"📦 Procesando videos de '{label}' en {input_path}...")

    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi'))]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for video_file in video_files:
            executor.submit(
                process_single_video,
                video_file,
                input_path,
                output_path,
                buffer_size,
                overlap,
                target_size,
                label,
                augment  # <-- pasa el parámetro
            )

def process_single_video(video_file, input_path, output_path, buffer_size=16, overlap=0.3, target_size=(224, 224), label=None, augment=False):
    video_path = os.path.join(input_path, video_file)
    temp_dir = None

    try:
        # Si es .avi, conviértelo temporalmente a .mp4
        if video_file.lower().endswith('.avi'):
            video_path, temp_dir = convertir_avi_a_mp4_temporal(video_path)

        frames = load_video_frames(video_path)
        buffers = create_buffers(frames, buffer_size, overlap, target_size, augment=augment)

        for idx, buffer in enumerate(buffers):
            save_path = os.path.join(output_path, f'{os.path.splitext(video_file)[0]}_buffer_{idx}.npy')
            np.save(save_path, buffer)

        print(f'✅ Procesado: {video_file} → {len(buffers)} buffers generados')

    except Exception as e:
        print(f'❌ Error procesando {video_file}: {e}')
    finally:
        # Limpia el archivo temporal si se creó
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

