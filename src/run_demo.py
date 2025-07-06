import os
import sys
import numpy as np
from dotenv import load_dotenv
from keras.models import load_model
from datetime import datetime
import subprocess
import tempfile
import shutil
import cv2
import random
from src.config.paths import VIDEO_PATHS
from src.preprocessing.loader_opencv import load_video_frames
from src.preprocessing.buffer_creator import create_buffers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'keras', 'epoch_3', 'fall_detector_model.keras')
RESULTADOS_DIR = os.path.join(SCRIPT_DIR, 'resultados')
os.makedirs(RESULTADOS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 16
FRAME_SIZE = (224, 224)
THRESHOLD = 0.5
VOTE_RATIO = 0.6
OVERLAP = 0.1
PORCENTAJE = 0.1  # Solo 10%

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(RESULTADOS_DIR, f"demo_test_eval_{timestamp}.txt")
result_log = open(result_file, 'w', encoding='utf-8')

# ─── Utilidades ────────────────────────────────────────────────────────────────
def convertir_avi_a_mp4_temporal(ruta_avi):
    temp_dir = tempfile.mkdtemp()
    ruta_mp4 = os.path.join(temp_dir, os.path.splitext(os.path.basename(ruta_avi))[0] + ".mp4")
    comando = [
        'ffmpeg', '-y', '-i', ruta_avi,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'aac', '-b:a', '128k',
        ruta_mp4
    ]
    try:
        subprocess.run(comando, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ruta_mp4, temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Error convirtiendo {ruta_avi} a mp4: {e}")

def evaluar_video(video_path):
    temp_dir = None
    try:
        if video_path.lower().endswith('.avi'):
            video_path, temp_dir = convertir_avi_a_mp4_temporal(video_path)

        frames = load_video_frames(video_path)
        buffers = create_buffers(frames, buffer_size=SEQUENCE_LENGTH, overlap=OVERLAP, target_size=FRAME_SIZE)

        if not buffers:
            return "NO CAÍDA", 0.0, 0

        preds = []
        probs = []

        for buffer in buffers:
            x = np.expand_dims(buffer, axis=0)
            prob = float(model.predict(x, verbose=0).squeeze())
            pred = 1 if prob >= THRESHOLD else 0
            preds.append(pred)
            probs.append(prob)

        vote_frac = sum(preds) / len(preds)
        final = "CAÍDA" if vote_frac >= VOTE_RATIO else "NO CAÍDA"
        return final, float(np.mean(probs)), len(buffers)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def evaluar_dataset(path, etiqueta_real, porcentaje=PORCENTAJE):
    aciertos, total = 0, 0
    y_true, y_pred = [], []
    resultados = []

    video_files = [f for f in os.listdir(path) if f.endswith(('.mp4', '.avi'))]
    video_files = sorted(video_files)

    # Seleccionar aleatoriamente el 10%
    cantidad = max(1, int(len(video_files) * porcentaje))
    random.seed(42)
    seleccionados = random.sample(video_files, cantidad)

    for video_file in seleccionados:
        print(f"🎥 Evaluando video: {video_file}")
        full_path = os.path.join(path, video_file)
        pred, prob, n_buffers = evaluar_video(full_path)
        correcto = "✅" if pred == etiqueta_real else "❌"

        y_true.append(1 if etiqueta_real == "CAÍDA" else 0)
        y_pred.append(1 if pred == "CAÍDA" else 0)

        total += 1
        aciertos += 1 if correcto == "✅" else 0

        resultados.append(f"{video_file} ({etiqueta_real}) → {pred} | Prob: {prob:.2f} | Buffers: {n_buffers} {correcto}")

    return resultados, y_true, y_pred, aciertos, total

# ─── Main ──────────────────────────────────────────────────────────────────────
print(f"📥 Cargando modelo desde: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("❌ Modelo no encontrado.")
    sys.exit(1)
model = load_model(MODEL_PATH)

def main():
    caida_path = VIDEO_PATHS['test']['caida']
    no_caida_path = VIDEO_PATHS['test']['no_caida']

    if not caida_path or not no_caida_path:
        print("❌ ERROR: Rutas de TEST no definidas.")
        sys.exit(1)

    result_log.write("🧪 Evaluación del modelo con 10% del dataset TEST\n")
    result_log.write("="*60 + "\n")

    print("🔍 Procesando CAÍDA...")
    res_caida, yt_c, yp_c, acc_c, tot_c = evaluar_dataset(caida_path, "CAÍDA")

    print("🔍 Procesando NO CAÍDA...")
    res_nocaida, yt_nc, yp_nc, acc_nc, tot_nc = evaluar_dataset(no_caida_path, "NO CAÍDA")

    for r in res_caida + res_nocaida:
        result_log.write(r + "\n")

    y_true = yt_c + yt_nc
    y_pred = yp_c + yp_nc

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    result_log.write("\n📊 RESUMEN (10% TEST):\n")
    result_log.write(f"Aciertos CAÍDA:    {acc_c}/{tot_c}\n")
    result_log.write(f"Aciertos NO CAÍDA: {acc_nc}/{tot_nc}\n")
    result_log.write(f"Total:             {acc_c + acc_nc}/{tot_c + tot_nc}\n\n")

    result_log.write(f"Accuracy:  {acc:.4f}\n")
    result_log.write(f"Precision: {prec:.4f}\n")
    result_log.write(f"Recall:    {rec:.4f}\n")
    result_log.write(f"F1 Score:  {f1:.4f}\n")

    result_log.close()
    print(f"\n[✅] Resultados guardados en: {result_file}")

if __name__ == "__main__":
    main()
