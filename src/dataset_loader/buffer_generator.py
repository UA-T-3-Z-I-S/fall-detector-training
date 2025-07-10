import os
import numpy as np
from keras.utils import Sequence

class BufferGenerator(Sequence):
    def __init__(self, caida_dirs, no_caida_dirs, batch_size=12, shuffle=True, class_weights=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}

        self.files = []
        self.labels = []

        if isinstance(caida_dirs, str):
            caida_dirs = [caida_dirs]
        if isinstance(no_caida_dirs, str):
            no_caida_dirs = [no_caida_dirs]

        for d in caida_dirs:
            if d and os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith('.npy'):
                        self.files.append(os.path.join(d, f))
                        self.labels.append(1)

        for d in no_caida_dirs:
            if d and os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith('.npy'):
                        self.files.append(os.path.join(d, f))
                        self.labels.append(0)

        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = [self.files[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]

        batch_buffers = [np.load(f) for f in batch_files]
        batch_buffers = np.array(batch_buffers)
        batch_labels = np.array(batch_labels)

        sample_weights = np.array([self.class_weights[label] for label in batch_labels])

        return batch_buffers, batch_labels, sample_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

