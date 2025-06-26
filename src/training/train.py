import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os

from src.dataset_loader.buffer_loader import load_train_data, load_val_data, load_test_data
from src.models.cnn_model import build_cnn_model
from src.models.lstm_model import build_lstm_model
from src.utils.class_weights import get_class_weights


def train():
    # Cargar datos
    print("📦 Cargando datos de entrenamiento, validación y test...")
    X_train, y_train = load_train_data()
    X_val, y_val = load_val_data()
    X_test, y_test = load_test_data()

    # CNN
    print("🧠 Construyendo modelo CNN...")
    cnn = build_cnn_model()
    cnn.trainable = True

    print("📊 Extrayendo características...")
    X_train_feat = cnn.predict(X_train)
    X_val_feat = cnn.predict(X_val)
    X_test_feat = cnn.predict(X_test)

    # LSTM
    print("🧠 Construyendo modelo LSTM...")
    lstm = build_lstm_model(input_shape=X_train_feat.shape[1:])
    lstm.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    class_weights = get_class_weights()
    print(f"⚖️  Pesos de clase aplicados: {class_weights}")

    checkpoint = ModelCheckpoint('models/fall_detector_model.keras', monitor='val_accuracy', save_best_only=True)

    # Entrenamiento
    print("🚀 Iniciando entrenamiento...")
    lstm.fit(
        X_train_feat, y_train,
        validation_data=(X_val_feat, y_val),
        epochs=20,
        batch_size=8,
        class_weight=class_weights,
        callbacks=[checkpoint]
    )

    print("✅ Entrenamiento terminado.")

    # Evaluación con test
    print("🧪 Evaluando modelo...")
    y_pred_probs = lstm.predict(X_test_feat)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📈 Resultados en test:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Guardar métricas en TXT
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"models/results_test_{now}.txt"
    with open(results_path, "w") as f:
        f.write(f"Resultados del modelo ({now}):\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")

    # Guardar modelo con nombre personalizado
    model_name = f"models/model_acc{int(acc*100)}_f1{int(f1*100)}.keras"
    lstm.save(model_name)
    print(f"\n💾 Modelo guardado como: {model_name}")
    print(f"📝 Resultados guardados en: {results_path}")


if __name__ == "__main__":
    train()
