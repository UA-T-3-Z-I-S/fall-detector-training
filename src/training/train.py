import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from src.dataset_loader.buffer_loader import load_buffers
from src.models.cnn_model import build_cnn_model
from src.models.lstm_model import build_lstm_model

def train():
    # Cargar buffers
    print("Cargando buffers...")
    X, y = load_buffers()

    # Dividir dataset
    print("Dividiendo dataset en entrenamiento y validación...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Definir y construir el modelo CNN
    print("Construyendo modelo CNN...")
    cnn = build_cnn_model()
    cnn.trainable = True  # Permitimos entrenar las últimas capas

    # Extraer features con la CNN
    print("Extrayendo características...")
    X_train_features = cnn.predict(X_train)
    X_val_features = cnn.predict(X_val)

    # Definir el modelo LSTM
    print("Construyendo modelo LSTM...")
    lstm = build_lstm_model(input_shape=X_train_features.shape[1:])

    # Compilar modelo LSTM
    lstm.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Configurar checkpoint
    checkpoint = ModelCheckpoint('models/fall_detector_model.keras', monitor='val_accuracy', save_best_only=True)

    # Entrenar modelo LSTM
    print("Iniciando entrenamiento...")
    lstm.fit(
        X_train_features, y_train,
        validation_data=(X_val_features, y_val),
        epochs=20,
        batch_size=8,
        callbacks=[checkpoint]
    )

    print("✅ Entrenamiento finalizado. Modelo guardado en 'models/fall_detector_model.keras'.")

if __name__ == "__main__":
    train()
