from keras.layers import LSTM, Dense, Dropout

# Devuelve una función que aplica las capas, no un modelo Sequential
def build_lstm_layers(units=64, num_classes=2):
    def lstm_block(inputs):
        x = LSTM(units, return_sequences=False)(inputs)
        x = Dropout(0.5)(x)  # Dropout extra
        x = Dense(64, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)
        return x
    return lstm_block
