from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D, BatchNormalization
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
import numpy as np

def build_combined_model(input_shape=(16, 224, 224, 3), num_classes=2, cnn_trainable=True):
    # Entrada de secuencia de frames
    video_input = Input(shape=input_shape, name="video_input")

    # CNN base (EfficientNetB0) con más capas congeladas
    base_model = EfficientNetB0(include_top=False, weights="imagenet", pooling=None)
    for layer in base_model.layers[:int(len(base_model.layers) * 0.85)]:
        layer.trainable = False
    for layer in base_model.layers[int(len(base_model.layers) * 0.85):]:
        layer.trainable = cnn_trainable

    # Aplicar CNN a cada frame
    x = TimeDistributed(base_model)(video_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.6))(x)  # Más dropout

    # LSTM más pequeño
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.6)(x)

    # Capa densa más pequeña
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=video_input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

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
