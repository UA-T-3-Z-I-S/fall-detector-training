import numpy as np
from sklearn.utils import class_weight
from src.dataset_loader.buffer_loader import load_train_data

def get_class_weights():
    _, y_train = load_train_data()

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    return dict(enumerate(class_weights))
