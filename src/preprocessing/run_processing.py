from src.preprocessing.video_processor import process_videos
from src.config.paths import VIDEO_PATHS, BUFFER_PATHS

def run_processing():
    for subset in ['train', 'val', 'test']:
        for clase in ['caida', 'no_caida']:
            ruta_entrada = VIDEO_PATHS[subset][clase]
            ruta_salida = BUFFER_PATHS[subset][clase]
            print(f"Procesando {subset.upper()} - {clase}...")
            process_videos(ruta_entrada, ruta_salida, label=clase)
