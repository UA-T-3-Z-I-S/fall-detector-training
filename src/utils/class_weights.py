import os
import numpy as np
from sklearn.utils import class_weight
from src.config.paths import BUFFER_PATHS

def get_class_weights():
    caida_dir = BUFFER_PATHS['train']['caida']
    no_caida_dir = BUFFER_PATHS['train']['no_caida']

    # Contar archivos .npy en cada clase
    caida_files = [f for f in os.listdir(caida_dir) if f.endswith('.npy')]
    no_caida_files = [f for f in os.listdir(no_caida_dir) if f.endswith('.npy')]

    # Generar etiquetas (1 = caída, 0 = no caída)
    labels = [1] * len(caida_files) + [0] * len(no_caida_files)

    # Calcular pesos de clase
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    return dict(enumerate(weights))
