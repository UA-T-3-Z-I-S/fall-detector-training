import os
from dotenv import load_dotenv

load_dotenv()

def get_env_list(prefix, n=5):
    """Devuelve una lista de rutas de variables de entorno con prefijo y número."""
    return [os.getenv(f"{prefix}_{i}") for i in range(1, n+1) if os.getenv(f"{prefix}_{i}")]

# 🟦 VIDEOS (sin procesar)
VIDEO_PATHS = {
    'train': {
        'caida': get_env_list('DATASET_TRAIN_CAIDA'),
        'no_caida': get_env_list('DATASET_TRAIN_NO_CAIDA'),
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
        'caida': get_env_list('BUFFER_TRAIN_CAIDA'),
        'no_caida': get_env_list('BUFFER_TRAIN_NO_CAIDA')
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
