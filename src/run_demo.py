import os
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from dotenv import load_dotenv
from datetime import datetime
import random

# ─── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()
TEST_CAIDA    = os.getenv('DATASET_TEST_CAIDA')
TEST_NO_CAIDA = os.getenv('DATASET_TEST_NO_CAIDA')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fall_detector_model.keras')
model      = load_model(MODEL_PATH)

SEQUENCE_LENGTH = 16
FRAME_SIZE      = (224, 224)

RESULTADOS_DIR = os.path.join(SCRIPT_DIR, 'resultados')
os.makedirs(RESULTADOS_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(RESULTADOS_DIR, f"modelo_demo_{timestamp}.txt")
result_log  = open(result_file, 'w', encoding='utf-8')
# ───────────────────────────────────────────────────────────────────────────────

def extract_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frames.append(preprocess_input(frame.astype(np.float32)))
    cap.release()
    return np.array(frames)

def sliding_windows(frames, seq_len, overlap):
    step = max(1, int(seq_len * (1 - overlap)))
    for st in range(0, len(frames) - seq_len + 1, step):
        yield frames[st:st + seq_len]

def predict_window(window, threshold):
    x    = window[np.newaxis, ...]
    prob = float(model.predict(x, verbose=0)[0][1])
    return (1 if prob >= threshold else 0), prob

def evaluar_video(frames, threshold, overlap, vote_ratio):
    preds, probs = [], []
    for win in sliding_windows(frames, SEQUENCE_LENGTH, overlap):
        p, pr = predict_window(win, threshold)
        preds.append(p)
        probs.append(pr)
    if preds:
        frac = sum(preds) / len(preds)
        final = "CAÍDA" if frac >= vote_ratio else "NO CAÍDA"
        avg_p = float(np.mean(probs))
    else:
        final, avg_p = "NO CAÍDA", 0.0
    return final, avg_p

def get_sampled_videos(folder, porcentaje=0.1):
    videos = [f for f in os.listdir(folder) if f.endswith('.mp4')]
    n = max(1, int(len(videos) * porcentaje))
    return random.sample(videos, n)

def procesar_dataset(folder, label_real, threshold, overlap, vote_ratio, porcentaje=0.1):
    aciertos, total = 0, 0
    resultados = []

    sampled_videos = get_sampled_videos(folder, porcentaje)

    for vf in sorted(sampled_videos):
        video_path = os.path.join(folder, vf)
        frames = extract_all_frames(video_path)
        pred, prob = evaluar_video(frames, threshold, overlap, vote_ratio)

        correcto = "✅" if pred == label_real else "❌"
        if correcto == "✅":
            aciertos += 1
        total += 1

        resultados.append(f"{vf} ({label_real}) -> {pred} ({prob:.2f}) {correcto}")

    return resultados, aciertos, total

def main():
    #thresholds = [0.5, 0.7, 0.9]
    #thresholds = [0.5]
    thresholds = [0.7]
    #thresholds = [0.9]

    #overlaps   = [0.3, 0.1, 0.0]
    overlaps   = [0.3]
    #overlaps   = [0.2]
    #overlaps   = [0.1]
    #overlaps   = [0.0]

    #vote_ratio = 0.5
    vote_ratio = 0.6
    #vote_ratio = 0.7
    #vote_ratio = 0.85

    for overlap in overlaps:
        for threshold in thresholds:
            result_log.write("\n" + "="*60 + "\n")
            result_log.write(f"🔍 Evaluación: OVERLAP = {overlap:.2f}, THRESHOLD = {threshold:.2f}\n")
            result_log.write("="*60 + "\n")

            # Procesar CAÍDA
            caida_res, caida_aciertos, caida_total = procesar_dataset(
                TEST_CAIDA, "CAÍDA", threshold, overlap, vote_ratio)

            # Procesar NO CAÍDA
            no_caida_res, no_caida_aciertos, no_caida_total = procesar_dataset(
                TEST_NO_CAIDA, "NO CAÍDA", threshold, overlap, vote_ratio)

            # Escribir resultados individuales
            for r in caida_res + no_caida_res:
                result_log.write(r + "\n")

            # Resumen por combinación
            total_aciertos = caida_aciertos + no_caida_aciertos
            total_videos   = caida_total + no_caida_total

            result_log.write("\n📊 RESUMEN:\n")
            result_log.write(f"Aciertos en CAÍDA:    {caida_aciertos}/{caida_total} = {caida_aciertos/caida_total:.2%}\n")
            result_log.write(f"Aciertos en NO CAÍDA: {no_caida_aciertos}/{no_caida_total} = {no_caida_aciertos/no_caida_total:.2%}\n")
            result_log.write(f"TOTAL GLOBAL:         {total_aciertos}/{total_videos} = {total_aciertos/total_videos:.2%}\n")
            result_log.write("\n\n")
            result_log.flush()

    result_log.close()
    print(f"[✅] Resultados guardados en: {result_file}")

if __name__ == "__main__":
    main()
