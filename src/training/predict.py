import numpy as np
from keras.models import load_model
from src.models.cnn_model import build_cnn_model

def predict(buffer_path, model_path='models/fall_detector_model.keras'):
    # Cargar modelo LSTM entrenado
    lstm = load_model(model_path)

    # Cargar CNN para extracción de características
    cnn = build_cnn_model()

    # Cargar buffer
    buffer = np.load(buffer_path)
    buffer = np.expand_dims(buffer, axis=0)  # (1, 16, 224, 224, 3)

    # Extraer características
    features = cnn.predict(buffer)

    # Realizar predicción
    prediction = lstm.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return 'Caída' if predicted_class == 1 else 'No Caída'
