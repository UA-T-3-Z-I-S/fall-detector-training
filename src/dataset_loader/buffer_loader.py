import os
import numpy as np
from src.config.paths import BUFFER_CAIDA, BUFFER_NO_CAIDA

def load_buffers(buffer_caida_path=BUFFER_CAIDA, buffer_no_caida_path=BUFFER_NO_CAIDA):
    buffers = []
    labels = []

    # Cargar buffers de caídas (etiqueta 1)
    for file in os.listdir(buffer_caida_path):
        if file.endswith('.npy'):
            buffer = np.load(os.path.join(buffer_caida_path, file))
            buffers.append(buffer)
            labels.append(1)  # 1 = caída

    # Cargar buffers de no caídas (etiqueta 0)
    for file in os.listdir(buffer_no_caida_path):
        if file.endswith('.npy'):
            buffer = np.load(os.path.join(buffer_no_caida_path, file))
            buffers.append(buffer)
            labels.append(0)  # 0 = no caída

    # Convertir a numpy arrays
    X = np.array(buffers)
    y = np.array(labels)

    return X, y
