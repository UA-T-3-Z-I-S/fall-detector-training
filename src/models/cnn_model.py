from keras.applications import EfficientNetB0
from keras.layers import TimeDistributed, GlobalAveragePooling2D
from keras.models import Sequential

def build_cnn_model(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar las dos primeras capas
    for layer in base_model.layers[:len(base_model.layers) // 3 * 2]:
        layer.trainable = False

    # Extraer características con Global Average Pooling
    model = Sequential([
        TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
        TimeDistributed(GlobalAveragePooling2D())
    ])

    return model
