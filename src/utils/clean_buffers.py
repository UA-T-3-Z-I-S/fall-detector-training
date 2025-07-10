import os
from src.config.paths import BUFFER_PATHS

def clean_buffers():
    for split in ['train', 'val', 'test']:
        for clase in ['caida', 'no_caida']:
            dirs = BUFFER_PATHS[split][clase]
            # Puede ser lista o string
            if isinstance(dirs, str):
                dirs = [dirs]
            for d in dirs:
                if d and os.path.exists(d):
                    files = [f for f in os.listdir(d) if f.endswith('.npy')]
                    for f in files:
                        try:
                            os.remove(os.path.join(d, f))
                        except Exception as e:
                            print(f"❌ Error borrando {f}: {e}")
                    print(f"🧹 Borrados {len(files)} buffers en {d}")

if __name__ == "__main__":
    clean_buffers()