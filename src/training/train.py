import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os

from src.models.combined_model import build_combined_model
from src.utils.class_weights import get_class_weights
from src.dataset_loader.buffer_generator import BufferGenerator
from src.config.paths import BUFFER_PATHS

def train():
    print("📦 Preparando generadores...")
    batch_size = 12
    train_gen = BufferGenerator(BUFFER_PATHS['train']['caida'], BUFFER_PATHS['train']['no_caida'], batch_size=batch_size)
    val_gen = BufferGenerator(BUFFER_PATHS['val']['caida'], BUFFER_PATHS['val']['no_caida'], batch_size=batch_size)
    test_gen = BufferGenerator(BUFFER_PATHS['test']['caida'], BUFFER_PATHS['test']['no_caida'], batch_size=batch_size, shuffle=False)

    # 🔍 Probar un batch para verificar
    print("🧪 Probando un batch del generador...")
    X_batch, y_batch = train_gen[0]
    print("Forma de X:", X_batch.shape)
    print("Forma de y:", y_batch.shape)

    # Modelo combinado
    print("🧠 Construyendo modelo combinado CNN + LSTM...")
    model = build_combined_model(input_shape=(16, 224, 224, 3), cnn_trainable=True)

    # Pesos de clase
    class_weights = get_class_weights()
    print(f"⚖️  Pesos de clase aplicados: {class_weights}")

    # Callbacks
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = ModelCheckpoint('models/fall_detector_model.keras', monitor='val_accuracy', save_best_only=True)
    csv_logger = CSVLogger(f"models/training_log_{now}.csv")

    # Entrenamiento
    print("🚀 Iniciando entrenamiento...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=class_weights,
        callbacks=[checkpoint, csv_logger]
    )
    print("✅ Entrenamiento finalizado.")

    # Evaluación
    print("🧪 Evaluando modelo en test...")
    y_true, y_pred = [], []
    for X_batch, y_batch in test_gen:
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

    # Guardar resultados
    results_path = f"models/results_test_{now}.txt"
    with open(results_path, "w") as f:
        f.write(f"Resultados del modelo ({now}):\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")

    # Guardar modelo final con nombre basado en desempeño
    model_name = f"models/model_acc{int(acc*100)}_f1{int(f1*100)}.keras"
    model.save(model_name)
    print(f"\n💾 Modelo guardado como: {model_name}")
    print(f"📝 Resultados guardados en: {results_path}")
    print(f"📄 Métricas por época: models/training_log_{now}.csv")


if __name__ == "__main__":
    train()
