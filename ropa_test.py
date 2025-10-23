#!/usr/bin/env python3
"""
ropa_test.py

Uso:
    python3 ropa_test.py

Requisitos:
    pip install tensorflow opencv-python numpy
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Ruta del modelo
MODEL_PATH = "/home/emerson/Documentos/tarea7/modelo_clasificacion_ropa.h5"
CLASSES_TXT = os.path.join(os.path.dirname(MODEL_PATH), "classes.txt")  # opcional

# Cargar el modelo
print("Cargando modelo desde:", MODEL_PATH)
model = load_model(MODEL_PATH)
model.summary()

# Cargar nombres de clases si existe classes.txt
if os.path.exists(CLASSES_TXT):
    with open(CLASSES_TXT, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Cargadas {len(classes)} clases desde {CLASSES_TXT}")
else:
    # Si no hay archivo, crear etiquetas genéricas
    n_outputs = model.output_shape[-1]
    classes = [f"class_{i}" for i in range(n_outputs)]
    print(f"No se encontró classes.txt, se crean {n_outputs} etiquetas genéricas.")

# --- Función de preprocesamiento (corregida para 28x28 en escala de grises) ---
def preprocess_image(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (28, 28))
    x = img_resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)  # (28,28,1)
    x = np.expand_dims(x, axis=0)   # (1,28,28,1)
    return x
# ------------------------------------------------------------------------------

# Abrir la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara. Verifica permisos o índice del dispositivo.")

win_name = "Clasificador de Ropa - 'q' para salir, 's' para snapshot"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Variables para FPS
prev_time = time.time()
frame_count = 0
fps = 0

snap_dir = Path("snapshots")
snap_dir.mkdir(exist_ok=True)

print("Iniciando cámara. Espera...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame vacío, intentando nuevamente...")
            time.sleep(0.1)
            continue

        frame_count += 1
        cur_time = time.time()
        if cur_time - prev_time >= 1.0:
            fps = frame_count / (cur_time - prev_time)
            prev_time = cur_time
            frame_count = 0

        # Preprocesar y predecir
        x = preprocess_image(frame)
        preds = model.predict(x, verbose=0)
        probs = preds[0] if preds.ndim == 2 else preds.flatten()

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = classes[top_idx] if top_idx < len(classes) else f"idx_{top_idx}"

        # Mostrar etiqueta y probabilidad en la imagen
        text = f"{label}: {top_prob*100:.1f}%  FPS:{fps:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        pad = 6
        cv2.rectangle(frame, (5, 5), (5 + text_w + pad, 5 + text_h + pad), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 5 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fname = snap_dir / f"snapshot_{timestamp}.jpg"
            cv2.imwrite(str(fname), frame)
            print("Snapshot guardado en:", fname)

except KeyboardInterrupt:
    print("Interrumpido por teclado.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada. Fin.")
