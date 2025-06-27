import numpy as np
import os
import time
from datetime import datetime
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.combined_model import build_combined_model
from src.utils.class_weights import get_class_weights
from src.dataset_loader.buffer_generator import BufferGenerator
from src.config.paths import BUFFER_PATHS
from src.utils.tqdm_callback import TQDMProgressBar

def train(epochs_to_train=2):
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

    test_gen = BufferGenerator(
        BUFFER_PATHS['test']['caida'],
        BUFFER_PATHS['test']['no_caida'],
        batch_size=batch_size,
        shuffle=False
    )

    print("🧪 Probando un batch del generador...")
    X_batch, y_batch, _ = train_gen[0]
    print("✅ Batch cargado correctamente.")
    print("Forma de X:", X_batch.shape)

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

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=False)
    csv_logger = CSVLogger(csv_path, append=True)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    progress_bar = TQDMProgressBar(update_every=10)

    start_time = time.time()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=final_epoch,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint, csv_logger, progress_bar, early_stopping],
        verbose=0
    )
    print(f"✅ Entrenamiento finalizado en {int((time.time() - start_time) / 60)} minutos.")

    with open(epoch_log_path, "w") as f:
        f.write(str(final_epoch))

    print("🧪 Evaluando modelo en test...")
    y_true, y_pred = [], []
    for i, (X_batch, y_batch) in enumerate(test_gen):
        print(f"🔢 Evaluando batch {i+1}/{len(test_gen)}...")
        y_probs = model.predict(X_batch)
        preds = np.argmax(y_probs, axis=1)
        y_true.extend(y_batch)
        y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n📈 Resultados en test:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    results_path = os.path.join(models_dir, "results_test.txt")
    with open(results_path, "w") as f:
        f.write("Resultados del modelo:\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")

    model_name = os.path.join(models_dir, f"model_acc{int(acc*100)}_f1{int(f1*100)}.keras")
    model.save(model_name)
    print(f"\n💾 Modelo guardado como: {model_name}")
    print(f"📝 Resultados guardados en: {results_path}")
    print(f"📄 Métricas por época: {csv_path}")
    print(f"🧾 Epochs acumuladas: {final_epoch} (guardadas en {epoch_log_path})")

if __name__ == "__main__":
    train(epochs_to_train=2)
