import numpy as np
import os
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.dataset_loader.buffer_generator import BufferGenerator
from src.config.paths import BUFFER_PATHS

def evaluate():
    print("📦 Preparando test generator...")
    batch_size = 12

    test_gen = BufferGenerator(
        BUFFER_PATHS['test']['caida'],
        BUFFER_PATHS['test']['no_caida'],
        batch_size=batch_size,
        shuffle=False
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "fall_detector_model.keras")

    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado en: {model_path}")
        return

    print(f"📥 Cargando modelo desde {model_path}...")
    model = load_model(model_path)

    print("🧪 Evaluando modelo en test...")
    y_true, y_pred = [], []

    for i in range(len(test_gen)):
        X_batch, y_batch, _ = test_gen[i]

        if X_batch is None or len(X_batch) == 0:
            print(f"⚠️ Batch {i + 1} vacío, saltando...")
            continue

        print(f"🔢 Evaluando batch {i + 1}/{len(test_gen)}...")
        try:
            y_probs = model.predict(X_batch, verbose=0)
        except Exception as e:
            print(f"❌ Error al predecir el batch {i + 1}: {e}")
            continue

        preds = np.argmax(y_probs, axis=1)
        y_true.extend(y_batch)
        y_pred.extend(preds)

    if not y_true:
        print("❌ No se pudo evaluar: y_true vacío.")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

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

if __name__ == "__main__":
    evaluate()
