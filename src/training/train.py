import numpy as np
import os
import time
from datetime import datetime
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback

from src.models.combined_model import build_combined_model
from src.utils.class_weights import get_class_weights
from src.dataset_loader.buffer_generator import BufferGenerator
from src.config.paths import BUFFER_PATHS
from src.utils.tqdm_callback import TQDMProgressBar

class CustomStopCallback(Callback):
    def __init__(self, threshold=0.82, patience=3, min_improvement=0.01):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.min_improvement = min_improvement
        self.wait = 0
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc is None:
            return

        if val_acc < self.threshold:
            print(f"⚠️  val_accuracy ({val_acc:.4f}) < umbral {self.threshold:.2f}")

        improvement = val_acc - self.best
        if improvement >= self.min_improvement:
            self.best = val_acc
            self.wait = 0
            print(f"✅ Mejora detectada en val_accuracy: {val_acc:.4f}")
        else:
            self.wait += 1
            print(f"⏳ No hubo mejora suficiente ({self.wait}/{self.patience})")
            if self.wait >= self.patience and val_acc < self.threshold:
                print("🛑 Parando entrenamiento: val_accuracy bajo y sin mejoras suficientes.")
                self.model.stop_training = True

def train(epochs_to_train=20):
    print("📦 Preparando generadores...")
    batch_size = 12
    class_weights = get_class_weights()
    print(f"⚖️  Pesos de clase aplicados: {class_weights}")

    train_gen = BufferGenerator(
        BUFFER_PATHS['train']['caida'],
        BUFFER_PATHS['train']['no_caida'],
        batch_size=batch_size,
        class_weights=class_weights
    )

    val_gen = BufferGenerator(
        BUFFER_PATHS['val']['caida'],
        BUFFER_PATHS['val']['no_caida'],
        batch_size=batch_size,
        class_weights=class_weights
    )

    print("🧪 Probando un batch del generador...")
    X_batch, y_batch, sample_weights = train_gen[0]
    print("✅ Batch cargado correctamente.")
    print("Forma de X:", X_batch.shape)
    print("🎯 Sample weights (primeros):", sample_weights[:5])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "fall_detector_model.keras")
    epoch_log_path = os.path.join(models_dir, "epoch_log.txt")
    csv_path = os.path.join(models_dir, "training_log.csv")

    initial_epoch = 0
    if os.path.exists(epoch_log_path):
        with open(epoch_log_path, "r") as f:
            initial_epoch = int(f.read().strip())

    if os.path.exists(model_path):
        print(f"♻️ Cargando modelo previo desde {model_path}...")
        model = load_model(model_path)
    else:
        print("🧠 Construyendo modelo nuevo CNN + LSTM...")
        model = build_combined_model(input_shape=(16, 224, 224, 3), cnn_trainable=True)
        print("✅ Modelo construido.")

    final_epoch = initial_epoch + epochs_to_train
    print(f"🚀 Entrenando desde epoch {initial_epoch + 1} hasta {final_epoch}...")

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    csv_logger = CSVLogger(csv_path, append=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
    progress_bar = TQDMProgressBar(update_every=10)
    custom_stop = CustomStopCallback(threshold=0.82, patience=3, min_improvement=0.01)
    start_time = time.time()
    try:
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=final_epoch,
            initial_epoch=initial_epoch,
            verbose=0,
            callbacks=[checkpoint, csv_logger, early_stopping, progress_bar, custom_stop]
        )
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento interrumpido manualmente. Guardando progreso...")

    print(f"✅ Entrenamiento finalizado en {int((time.time() - start_time) / 60)} minutos.")
    with open(epoch_log_path, "w") as f:
        f.write(str(final_epoch))

def train_etapas():
    etapas = [1, 2, 3, 4, 5]
    epocas_por_etapa = [1, 2, 3, 3, 3]  # Puedes ajustar esto si lo deseas

    batch_size = 12
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "fall_detector_model.keras")
    epoch_log_path = os.path.join(models_dir, "epoch_log.txt")
    csv_path = os.path.join(models_dir, "training_log.csv")

    initial_epoch = 0
    if os.path.exists(epoch_log_path):
        with open(epoch_log_path, "r") as f:
            initial_epoch = int(f.read().strip())

    if os.path.exists(model_path):
        print(f"♻️ Cargando modelo previo desde {model_path}...")
        model = load_model(model_path)
    else:
        print("🧠 Construyendo modelo nuevo CNN + LSTM...")
        model = build_combined_model(input_shape=(16, 224, 224, 3), cnn_trainable=True)
        print("✅ Modelo construido.")

    val_gen = BufferGenerator(
        BUFFER_PATHS['val']['caida'],
        BUFFER_PATHS['val']['no_caida'],
        batch_size=batch_size,
        class_weights={0: 1.0, 1: 1.0}
    )

    for idx, etapa in enumerate(etapas):
        print(f"\n=== Entrenando ETAPA {etapa} ({epocas_por_etapa[idx]} épocas) ===")
        caida_dirs = [BUFFER_PATHS['train']['caida'][etapa-1]]
        no_caida_dirs = [BUFFER_PATHS['train']['no_caida'][etapa-1]]

        if not all(os.path.exists(d) for d in caida_dirs + no_caida_dirs):
            print(f"❌ Alguna carpeta de la etapa {etapa} no existe. Saltando...")
            continue

        # Calcula pesos SOLO para esta etapa
        def contar_npy(dirs):
            total = 0
            for d in dirs:
                if d and os.path.exists(d):
                    total += len([f for f in os.listdir(d) if f.endswith('.npy')])
            return total

        n_caida = contar_npy(caida_dirs)
        n_no_caida = contar_npy(no_caida_dirs)
        labels = [1]*n_caida + [0]*n_no_caida
        if labels:
            from sklearn.utils import class_weight
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            class_weights = dict(enumerate(weights))
        else:
            class_weights = {0: 1.0, 1: 1.0}
        print(f"⚖️  Pesos de clase para etapa {etapa}: {class_weights}")

        train_gen = BufferGenerator(
            caida_dirs,
            no_caida_dirs,
            batch_size=batch_size,
            class_weights=class_weights
        )

        total = n_caida + n_no_caida
        if total > 0:
            print(f"🔢 Caída: {n_caida} ({n_caida/total:.1%}) | No caída: {n_no_caida} ({n_no_caida/total:.1%})")
        else:
            print("⚠️  No hay datos en esta etapa.")

        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
        csv_logger = CSVLogger(csv_path, append=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
        progress_bar = TQDMProgressBar(update_every=10)
        custom_stop = CustomStopCallback(threshold=0.82, patience=3, min_improvement=0.01)
        start_time = time.time()
        try:
            epocas = epocas_por_etapa[idx]
            final_epoch = initial_epoch + epocas
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=final_epoch,
                initial_epoch=initial_epoch,
                verbose=0,
                callbacks=[checkpoint, csv_logger, early_stopping, progress_bar, custom_stop]
            )
        except KeyboardInterrupt:
            print("\n🛑 Entrenamiento interrumpido manualmente. Guardando progreso...")
            break

        print(f"✅ Entrenamiento finalizado en {int((time.time() - start_time) / 60)} minutos.")
        with open(epoch_log_path, "w") as f:
            f.write(str(final_epoch))
        with open("stage_log.txt", "w") as f:
            f.write(str(etapa))
        initial_epoch = final_epoch  # Para la siguiente etapa

if __name__ == "__main__":
    train_etapas()
