import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

from src.dataset_loader.buffer_loader import load_test_data
from src.models.cnn_model import build_cnn_model

def evaluate(model_path='models/fall_detector_model.keras'):
    print("🧪 Evaluando modelo:", model_path)

    # Cargar modelo
    lstm = load_model(model_path)
    cnn = build_cnn_model()

    # Cargar buffers de test
    X_test, y_test = load_test_data()
    X_test_feat = cnn.predict(X_test)

    y_pred_probs = lstm.predict(X_test_feat)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📈 Resultados:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"models/results_test_{now}.txt"
    with open(results_path, "w") as f:
        f.write(f"Resultados del modelo ({now}):\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")

    print(f"\n📝 Guardado en: {results_path}")

if __name__ == "__main__":
    evaluate()
