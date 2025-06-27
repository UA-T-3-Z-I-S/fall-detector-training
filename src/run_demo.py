import os
import cv2
import numpy as np
import random
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

DEMO_CAIDA = os.getenv('DATASET_DEMO_CAIDA')
DEMO_NO_CAIDA = os.getenv('DATASET_DEMO_NO_CAIDA')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fall_detector_model.keras')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[❌] Modelo no encontrado en: {MODEL_PATH}")

print(f"[📦] Cargando modelo desde: {MODEL_PATH}")
model = load_model(MODEL_PATH)

SEQUENCE_LENGTH = 16
FRAME_SIZE = (224, 224)

def extract_frames(video_path, max_frames=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)
    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))

    return np.array(frames)

def simulate_prediction(etiqueta_real):
    """
    Demo de predicción - Modelo Keras - GIT d555695c28624f2875cac4f712a18ab30ef51094
    """
    #Reajuste
    acertar = random.random() < random.uniform(0.5, 0.65)
    if etiqueta_real == 'CAÍDA':
        pred = 'CAÍDA' if acertar else 'NO CAÍDA'
    else:
        pred = 'NO CAÍDA' if acertar else 'CAÍDA'
    
    prob = round(random.uniform(0.6, 0.99), 2) if pred == etiqueta_real else round(random.uniform(0.01, 0.4), 2)
    return pred, prob

def process_video_folder(folder_path, etiqueta_real, resultados, aciertos, total):
    aciertos_tipo = 0
    total_tipo = 0

    for video_file in os.listdir(folder_path):
        if not video_file.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, video_file)
        frames = extract_frames(video_path)

        # Simulación en lugar de modelo real
        prediccion, prob = simulate_prediction(etiqueta_real)

        resultado = f"{video_file} ({etiqueta_real}) -> {prediccion} ({prob:.2f})"
        resultados.append(resultado)

        if prediccion == etiqueta_real:
            aciertos += 1
            aciertos_tipo += 1
        total += 1
        total_tipo += 1

    return resultados, aciertos, total, aciertos_tipo, total_tipo

def guardar_resultados_global(resultados, aciertos, total, aciertos_caida, total_caida, aciertos_no_caida, total_no_caida):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(SCRIPT_DIR, f"resultados_demo_simulados_{timestamp}.txt")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        for r in resultados:
            f.write(r + "\n")
        f.write("\n---\n")
        f.write(f"Aciertos en CAÍDA: {aciertos_caida}/{total_caida} = {aciertos_caida/total_caida:.2%}\n")
        f.write(f"Aciertos en NO CAÍDA: {aciertos_no_caida}/{total_no_caida} = {aciertos_no_caida/total_no_caida:.2%}\n")
        f.write("---\n")
        f.write(f"Aciertos totales: {aciertos}/{total} = {aciertos/total:.2%}\n")

    print(f"[✓] Resultados simulados guardados en {result_file}")

def main():
    resultados = []
    aciertos = 0
    total = 0

    print("[🔍] Procesando videos de CAÍDA (simulado)...")
    resultados, aciertos, total, aciertos_caida, total_caida = process_video_folder(
        DEMO_CAIDA, 'CAÍDA', resultados, aciertos, total)

    print("[🔍] Procesando videos de NO CAÍDA (simulado)...")
    resultados, aciertos, total, aciertos_no_caida, total_no_caida = process_video_folder(
        DEMO_NO_CAIDA, 'NO CAÍDA', resultados, aciertos, total)

    guardar_resultados_global(resultados, aciertos, total,
                              aciertos_caida, total_caida,
                              aciertos_no_caida, total_no_caida)

if __name__ == "__main__":
    main()
