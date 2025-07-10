from keras.applications import EfficientNetB0
from keras.layers import TimeDistributed, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Sequential

def build_cnn_model(input_shape=(224, 224, 3), regularize=True):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar 2/3 del modelo
    for layer in base_model.layers[:len(base_model.layers) * 2 // 3]:
        layer.trainable = False

    layers = [
        TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
        TimeDistributed(GlobalAveragePooling2D())
    ]

    if regularize:
        layers.extend([
            TimeDistributed(BatchNormalization()),
            TimeDistributed(Dropout(0.5))  # Subido de 0.3 a 0.5
        ])

    return Sequential(layers)
