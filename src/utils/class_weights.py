import os
import numpy as np
from sklearn.utils import class_weight
from src.config.paths import BUFFER_PATHS

def get_class_weights():
    caida_dirs = BUFFER_PATHS['train']['caida']
    no_caida_dirs = BUFFER_PATHS['train']['no_caida']

    caida_files = []
    for d in caida_dirs:
        if d and os.path.exists(d):
            caida_files += [f for f in os.listdir(d) if f.endswith('.npy')]
    no_caida_files = []
    for d in no_caida_dirs:
        if d and os.path.exists(d):
            no_caida_files += [f for f in os.listdir(d) if f.endswith('.npy')]

    labels = [1] * len(caida_files) + [0] * len(no_caida_files)

    if not labels:
        return {0: 1.0, 1: 1.0}

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    return dict(enumerate(weights))
