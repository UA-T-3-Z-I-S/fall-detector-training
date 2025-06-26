import numpy as np
from sklearn.utils import class_weight
from src.dataset_loader.buffer_generator import BufferGenerator
from src.config.paths import BUFFER_PATHS

def get_class_weights():
    generator = BufferGenerator(
        caida_path=BUFFER_PATHS['train']['caida'],
        no_caida_path=BUFFER_PATHS['train']['no_caida'],
        batch_size=64,  # puede ser grande ya que solo usamos etiquetas
        shuffle=False
    )

    # Recolectar todas las etiquetas de entrenamiento
    all_labels = []
    for _, labels in generator:
        all_labels.extend(labels)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )

    return dict(enumerate(class_weights))
