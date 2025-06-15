import os
from dotenv import load_dotenv

load_dotenv()

# Paths de los videos originales
DATASET_CAIDA = os.getenv('DATASET_CAIDA')
DATASET_NO_CAIDA = os.getenv('DATASET_NO_CAIDA')

# Paths de los buffers procesados
BUFFER_CAIDA = os.getenv('BUFFER_CAIDA')
BUFFER_NO_CAIDA = os.getenv('BUFFER_NO_CAIDA')
