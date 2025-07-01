from src.preprocessing.video_processor import process_videos
from src.config.paths import VIDEO_PATHS, BUFFER_PATHS

def run_processing():
    for subset in ['train', 'val', 'test']:
        for clase in ['caida', 'no_caida']:
            rutas_entrada = VIDEO_PATHS[subset][clase]
            rutas_salida = BUFFER_PATHS[subset][clase]
            # Soporta lista o string
            if isinstance(rutas_entrada, str):
                rutas_entrada = [rutas_entrada]
            if isinstance(rutas_salida, str):
                rutas_salida = [rutas_salida]
            for ruta_entrada, ruta_salida in zip(rutas_entrada, rutas_salida):
                print(f"Procesando {subset.upper()} - {clase} - {ruta_entrada}...")
                augment = (subset == 'train')
                process_videos(ruta_entrada, ruta_salida, label=clase, augment=augment)
