from keras.models import Model
from keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, BatchNormalization, Dropout, LSTM, Dense
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
import numpy as np

def build_combined_model(input_shape=(16, 224, 224, 3), cnn_trainable=True):
    video_input = Input(shape=input_shape, name="video_input")
    base_model = EfficientNetB0(include_top=False, weights="imagenet", pooling=None)
    # Congela solo el 30% de las capas (descongela el 70%)
    for layer in base_model.layers[:int(len(base_model.layers) * 0.3)]:
        layer.trainable = False
    for layer in base_model.layers[int(len(base_model.layers) * 0.3):]:
        layer.trainable = cnn_trainable

    x = TimeDistributed(base_model)(video_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.4))(x)

    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=video_input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
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

    sample_weights = np.array([self.class_weights[label] for label in batch_labels])
    return batch_buffers, batch_labels, sample_weights
    # Devuelve sample_weight como parte del diccionario    return batch_buffers, batch_labels, sample_weights