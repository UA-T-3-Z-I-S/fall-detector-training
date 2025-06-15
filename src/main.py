import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.opencv_loader import process_videos
from src.config import paths

def main():
    print("Procesando videos de caídas...")
    process_videos(paths.DATASET_CAIDA, paths.BUFFER_CAIDA)

    print("\nProcesando videos sin caídas...")
    process_videos(paths.DATASET_NO_CAIDA, paths.BUFFER_NO_CAIDA)

    print("\nProcesamiento finalizado.")

if __name__ == "__main__":
    main()
