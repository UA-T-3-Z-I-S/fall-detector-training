from src.preprocessing.video_processor import process_videos
from src.config.paths import DATASET_CAIDA, DATASET_NO_CAIDA, BUFFER_CAIDA, BUFFER_NO_CAIDA

def run_processing():
    process_videos(DATASET_CAIDA, BUFFER_CAIDA, label='caídas')
    process_videos(DATASET_NO_CAIDA, BUFFER_NO_CAIDA, label='no caídas')
