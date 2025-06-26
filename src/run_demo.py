import os
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno desde .env
load_dotenv()

DEMO_CAIDA = os.getenv('DATASET_DEMO_CAIDA')
DEMO_NO_CAIDA = os.getenv('DATASET_DEMO_NO_CAIDA')
MODEL_PATH = os.path.join('models', 'fall_detector_model.keras')

SEQUENCE_LENGTH = 30
FRAME_SIZE = (224, 224)

model = load_model(MODEL_PATH)

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

def predict(frames):
    input_tensor = np.expand_dims(frames, axis=0)
    prob = model.predict(input_tensor)[0][0]
    return 'CAÍDA' if prob >= 0.5 else 'NO CAÍDA', prob

def process_video_folder(folder_path, etiqueta_real, resultados, aciertos, total):
    for video_file in os.listdir(folder_path):
        if not video_file.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, video_file)
        frames = extract_frames(video_path)
        prediccion, prob = predict(frames)

        resultado = f"{video_file} ({etiqueta_real}) -> {prediccion} ({prob:.2f})"
        resultados.append(resultado)

        if prediccion == etiqueta_real:
            aciertos += 1
        total += 1
    return resultados, aciertos, total

def guardar_resultados_global(resultados, aciertos, total):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"resultados_demo_{timestamp}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        for r in resultados:
            f.write(r + "\n")
        f.write("\n---\n")
        f.write(f"Aciertos totales: {aciertos}/{total} = {aciertos/total:.2%}\n")
    print(f"[✓] Resultados globales guardados en {result_file}")

def main():
    resultados = []
    aciertos = 0
    total = 0

    print("[🔍] Procesando videos de CAÍDA...")
    resultados, aciertos, total = process_video_folder(DEMO_CAIDA, 'CAÍDA', resultados, aciertos, total)

    print("[🔍] Procesando videos de NO CAÍDA...")
    resultados, aciertos, total = process_video_folder(DEMO_NO_CAIDA, 'NO CAÍDA', resultados, aciertos, total)

    guardar_resultados_global(resultados, aciertos, total)

if __name__ == "__main__":
    main()
