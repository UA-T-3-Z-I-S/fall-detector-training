import cv2

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames
