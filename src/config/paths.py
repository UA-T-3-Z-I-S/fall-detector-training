import os
from dotenv import load_dotenv

load_dotenv()

# 🟦 VIDEOS (sin procesar)
VIDEO_PATHS = {
    'train': {
        'caida': os.getenv('DATASET_TRAIN_CAIDA'),
        'no_caida': os.getenv('DATASET_TRAIN_NO_CAIDA'),
    },
    'val': {
        'caida': os.getenv('DATASET_VAL_CAIDA'),
        'no_caida': os.getenv('DATASET_VAL_NO_CAIDA'),
    },
    'test': {
        'caida': os.getenv('DATASET_TEST_CAIDA'),
        'no_caida': os.getenv('DATASET_TEST_NO_CAIDA'),
    },
    'demo': {
        'caida': os.getenv('DATASET_DEMO_CAIDA'),
        'no_caida': os.getenv('DATASET_DEMO_NO_CAIDA'),
    }
}

# 🟨 BUFFERS (procesados)
BUFFER_PATHS = {
    'train': {
        'caida': os.getenv('BUFFER_TRAIN_CAIDA'),
        'no_caida': os.getenv('BUFFER_TRAIN_NO_CAIDA'),
    },
    'val': {
        'caida': os.getenv('BUFFER_VAL_CAIDA'),
        'no_caida': os.getenv('BUFFER_VAL_NO_CAIDA'),
    },
    'test': {
        'caida': os.getenv('BUFFER_TEST_CAIDA'),
        'no_caida': os.getenv('BUFFER_TEST_NO_CAIDA'),
    }
}
