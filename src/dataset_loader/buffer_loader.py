import os
import numpy as np
from src.config.paths import (
    BUFFER_TRAIN_CAIDA, BUFFER_TRAIN_NO_CAIDA,
    BUFFER_VAL_CAIDA, BUFFER_VAL_NO_CAIDA,
    BUFFER_TEST_CAIDA, BUFFER_TEST_NO_CAIDA
)

def load_buffers(caida_path, no_caida_path):
    buffers = []
    labels = []

    # Cargar buffers de caídas (etiqueta 1)
    for file in os.listdir(caida_path):
        if file.endswith('.npy'):
            path = os.path.join(caida_path, file)
            buffer = np.load(path)
            buffers.append(buffer)
            labels.append(1)

    # Cargar buffers de no caídas (etiqueta 0)
    for file in os.listdir(no_caida_path):
        if file.endswith('.npy'):
            path = os.path.join(no_caida_path, file)
            buffer = np.load(path)
            buffers.append(buffer)
            labels.append(0)

    return np.array(buffers), np.array(labels)


# Interfaces específicas para cada dataset
def load_train_data():
    return load_buffers(BUFFER_TRAIN_CAIDA, BUFFER_TRAIN_NO_CAIDA)

def load_val_data():
    return load_buffers(BUFFER_VAL_CAIDA, BUFFER_VAL_NO_CAIDA)

def load_test_data():
    return load_buffers(BUFFER_TEST_CAIDA, BUFFER_TEST_NO_CAIDA)
