from keras.models import Model
from keras.layers import Input
from src.models.cnn_model import build_cnn_model
from src.models.lstm_model import build_lstm_layers  # Ahora solo devuelve las capas LSTM
from keras.optimizers import Adam
import numpy as np

def build_combined_model(input_shape=(16, 224, 224, 3), num_classes=2, cnn_trainable=True):
    # Entrada del modelo: secuencia de 16 frames RGB de 224x224
    inputs = Input(shape=input_shape)

    # Modelo CNN (preentrenado EfficientNetB0)
    cnn = build_cnn_model(input_shape=(224, 224, 3))
    cnn.trainable = cnn_trainable  # Puedes congelarlo o descongelarlo
    cnn_features = cnn(inputs)     # Shape: (batch, 16, feature_dim)

    # Modelo LSTM (recibe las features de cada frame)
    lstm_block = build_lstm_layers()
    lstm_output = lstm_block(cnn_features)  # Output: (batch, num_classes)

    # Modelo final
    model = Model(inputs=inputs, outputs=lstm_output)
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def __getitem__(self, idx):
    indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_files = [self.files[i] for i in indexes]
    batch_labels = [self.labels[i] for i in indexes]

    batch_buffers = [np.load(f) for f in batch_files]
    batch_buffers = np.array(batch_buffers)
    batch_labels = np.array(batch_labels)

    sample_weights = np.array([self.class_weights[label] for label in batch_labels])

    # Devuelve sample_weight como parte del diccionario
    return batch_buffers, batch_labels, sample_weights
