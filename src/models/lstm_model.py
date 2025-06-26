# src/models/lstm_model.py
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import layers

# Solo devuelve las capas, no el modelo completo
def build_lstm_layers(units=64, num_classes=2):
    return Sequential([
        LSTM(units, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
