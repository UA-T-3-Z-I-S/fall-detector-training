from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm_model(input_shape, num_classes=2):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
