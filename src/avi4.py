import os
import cv2
import subprocess

def convertir_avi_a_mp4(ruta_avi, ruta_mp4):
    comando = [
        'ffmpeg', '-y', '-i', ruta_avi,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'aac', '-b:a', '128k',
        ruta_mp4
    ]
    print(f"Convirtiendo {ruta_avi} a {ruta_mp4} ...")
    try:
        subprocess.run(comando, check=True)
        print(f"✅ Conversión exitosa: {ruta_mp4}")
    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")
        raise

# Cambia estas rutas por las que quieras probar
ruta_avi = r"C:\Users\sebas\OneDrive\Desktop\SISTEMA DE CAIDAS\datasets\videos\train\caida_4\multiple_chute01_cam1.avi"
ruta_mp4 = r"C:\Users\sebas\OneDrive\Desktop\SISTEMA DE CAIDAS\datasets\videos\train\caida_4\multiple_chute01_cam1.mp4"

if not os.path.exists(ruta_avi):
    print(f"❌ El archivo no existe: {ruta_avi}")
    exit(1)

convertir_avi_a_mp4(ruta_avi, ruta_mp4)

# Prueba cargar los frames del mp4 generado
if os.path.exists(ruta_mp4):
    cap = cv2.VideoCapture(ruta_mp4)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Frames leídos del mp4: {len(frames)}")
else:
    print(f"❌ El archivo mp4 no se generó correctamente.")