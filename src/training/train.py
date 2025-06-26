import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os
import time

from src.models.combined_model import build_combined_model
from src.utils.class_weights import get_class_weights
from src.dataset_loader.buffer_generator import BufferGenerator
from src.config.paths import BUFFER_PATHS
from src.utils.tqdm_callback import TQDMProgressBar

def train(epochs_to_train=2):
    print("📦 Preparando generadores...")
    batch_size = 12
    train_gen = BufferGenerator(BUFFER_PATHS['train']['caida'], BUFFER_PATHS['train']['no_caida'], batch_size=batch_size)
    val_gen = BufferGenerator(BUFFER_PATHS['val']['caida'], BUFFER_PATHS['val']['no_caida'], batch_size=batch_size)
    test_gen = BufferGenerator(BUFFER_PATHS['test']['caida'], BUFFER_PATHS['test']['no_caida'], batch_size=batch_size, shuffle=False)

    print("🧪 Probando un batch del generador...")
    X_batch, y_batch = train_gen[0]
    print("✅ Batch cargado correctamente.")
    print("Forma de X:", X_batch.shape)

    # 📁 Configurar paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "fall_detector_model.keras")
    epoch_log_path = os.path.join(models_dir, "epoch_log.txt")
    csv_path = os.path.join(models_dir, "training_log.csv")

    # 🔁 Epochs previos
    initial_epoch = 0
    if os.path.exists(epoch_log_path):
        with open(epoch_log_path, "r") as f:
            initial_epoch = int(f.read().strip())

    # 🧠 Cargar modelo anterior si existe
    if os.path.exists(model_path):
        print(f"♻️ Cargando modelo previo desde {model_path}...")
        model = load_model(model_path)
    else:
        print("🧠 Construyendo modelo nuevo CNN + LSTM...")
        model = build_combined_model(input_shape=(16, 224, 224, 3), cnn_trainable=True)
        print("✅ Modelo construido.")

    class_weights = get_class_weights()
    print(f"⚖️  Pesos de clase aplicados: {class_weights}")

    # 🧮 Epochs a entrenar
    final_epoch = initial_epoch + epochs_to_train
    print(f"🚀 Entrenando desde epoch {initial_epoch + 1} hasta {final_epoch}...")

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=False)
    csv_logger = CSVLogger(csv_path, append=True)

    start_time = time.time()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=final_epoch,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=[checkpoint, csv_logger, TQDMProgressBar()],
        verbose=0
    )
    print(f"✅ Entrenamiento finalizado en {int((time.time() - start_time) / 60)} minutos.")

    # 📝 Guardar nuevo número de épocas acumuladas
    with open(epoch_log_path, "w") as f:
        f.write(str(final_epoch))

    # 🧪 Evaluar modelo
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

    # 📄 Guardar resultados finales
    results_path = os.path.join(models_dir, f"results_test.txt")
    with open(results_path, "w") as f:
        f.write(f"Resultados del modelo:\n")
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
    train(epochs_to_train=2)  # Cambia aquí el número de nuevas épocas a entrenar
