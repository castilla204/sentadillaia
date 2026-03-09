import os
import subprocess
import sys
from typing import List, Tuple


def _ensure_package(pkg: str, import_name: str | None = None) -> None:
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        print(f"⚙️ Instalando dependencia faltante: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


_ensure_package("opencv-python", "cv2")
_ensure_package("numpy", "numpy")
_ensure_package("scikit-learn", "sklearn")
_ensure_package("joblib", "joblib")

import cv2
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# --- Manejo especial de mediapipe: si la versión instalada no tiene .solutions,
# forzamos instalar una versión compatible y la recargamos. ---
try:
    import mediapipe as mp  # type: ignore
    if not hasattr(mp, "solutions"):
        raise ImportError("mediapipe sin .solutions, reinstalando versión clásica...")
except Exception:
    print("⚙️ Ajustando versión de mediapipe para usar Pose...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe==0.10.14"])
    import importlib
    mp = importlib.import_module("mediapipe")  # type: ignore

from mediapipe import solutions as mp_solutions  # type: ignore
mp_pose = mp_solutions.pose

# Rutas base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
EJERCICIO = "sentadilla"
CLF_PATH = os.path.join(MODELOS_DIR, f"{EJERCICIO}_pose_bien_mal.joblib")

# Percentil usado para estimar el umbral de distancia Mahalanobis.
# Si es muy bajo, muchos vídeos "raros" se marcarán como fuera de distribución
# y se avisará de que no se parecen a las sentadillas de entrenamiento.
# Con un valor alto (por ejemplo 99.5) el detector es más permisivo.
OUTLIER_PERCENTIL = 99.5


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calcula el ángulo ABC (en grados) a partir de tres puntos 2D.
    """
    ba = a - b
    bc = c - b
    # Evitar divisiones por cero
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return 0.0
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)


def extract_pose_features(video_path: str, max_frames: int = 200) -> np.ndarray:
    """
    Extrae características específicas de sentadilla usando MediaPipe Pose:
    - Ángulos de rodilla (izq/der)
    - Ángulos de cadera (izq/der)
    - Ángulo del tronco (inclinación)
    Se devuelven estadísticas por vídeo: media, min, max y std de cada ángulo.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    angles_knee_l = []
    angles_knee_r = []
    angles_hip_l = []
    angles_hip_r = []
    angles_torso = []

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            frame_count += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark

            def p(idx):
                return np.array([lm[idx].x, lm[idx].y], dtype=np.float32)

            # Puntos clave
            hip_l = p(mp_pose.PoseLandmark.LEFT_HIP)
            knee_l = p(mp_pose.PoseLandmark.LEFT_KNEE)
            ankle_l = p(mp_pose.PoseLandmark.LEFT_ANKLE)

            hip_r = p(mp_pose.PoseLandmark.RIGHT_HIP)
            knee_r = p(mp_pose.PoseLandmark.RIGHT_KNEE)
            ankle_r = p(mp_pose.PoseLandmark.RIGHT_ANKLE)

            shoulder_l = p(mp_pose.PoseLandmark.LEFT_SHOULDER)
            shoulder_r = p(mp_pose.PoseLandmark.RIGHT_SHOULDER)

            # Ángulos de rodilla
            angles_knee_l.append(angle_3pts(hip_l, knee_l, ankle_l))
            angles_knee_r.append(angle_3pts(hip_r, knee_r, ankle_r))

            # Ángulos de cadera (rodilla-cadera-hombro)
            angles_hip_l.append(angle_3pts(knee_l, hip_l, shoulder_l))
            angles_hip_r.append(angle_3pts(knee_r, hip_r, shoulder_r))

            # Ángulo del tronco vs vertical (línea cadera-media de hombros)
            mid_shoulder = 0.5 * (shoulder_l + shoulder_r)
            vec_torso = mid_shoulder - 0.5 * (hip_l + hip_r)
            # Ángulo entre torso y eje vertical (0, -1)
            vertical = np.array([0.0, -1.0], dtype=np.float32)
            if np.linalg.norm(vec_torso) > 1e-6:
                cos_t = np.dot(vec_torso, vertical) / (np.linalg.norm(vec_torso) * np.linalg.norm(vertical))
                cos_t = np.clip(cos_t, -1.0, 1.0)
                ang_t = np.degrees(np.arccos(cos_t))
            else:
                ang_t = 0.0
            angles_torso.append(float(ang_t))

    cap.release()

    def stats(arr: List[float]) -> List[float]:
        if not arr:
            return [0.0, 0.0, 0.0, 0.0]
        a = np.array(arr, dtype=np.float32)
        return [float(a.mean()), float(a.min()), float(a.max()), float(a.std())]

    feat = []
    feat += stats(angles_knee_l)
    feat += stats(angles_knee_r)
    feat += stats(angles_hip_l)
    feat += stats(angles_hip_r)
    feat += stats(angles_torso)

    return np.array(feat, dtype=np.float32)


def listar_videos_etiquetados() -> Tuple[List[str], List[int]]:
    """
    Recorre:
        videos/sentadilla/bien  -> etiqueta 1
        videos/sentadilla/mal   -> etiqueta 0
    y devuelve listas X_paths, y_labels.
    """
    X_paths: List[str] = []
    y_labels: List[int] = []

    for etiqueta_nombre, etiqueta_valor in [("bien", 1), ("mal", 0)]:
        carpeta = os.path.join(VIDEOS_DIR, EJERCICIO, etiqueta_nombre)
        if not os.path.exists(carpeta):
            continue
        for f in os.listdir(carpeta):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                X_paths.append(os.path.join(carpeta, f))
                y_labels.append(etiqueta_valor)

    return X_paths, y_labels


def construir_dataset() -> Tuple[np.ndarray, np.ndarray]:
    print("📂 Buscando vídeos etiquetados en 'videos/sentadilla/bien' y 'videos/sentadilla/mal'...")
    X_paths, y = listar_videos_etiquetados()

    if len(X_paths) < 2:
        raise RuntimeError(
            "Necesito al menos 1 vídeo en 'bien' y 1 en 'mal' para poder entrenar.\n"
            "Pon tus propios vídeos de técnica buena en:\n"
            "  videos/sentadilla/bien\n"
            "y de técnica mala en:\n"
            "  videos/sentadilla/mal"
        )

    print(f"Encontrados {len(X_paths)} vídeos. Extrayendo características de pose (MediaPipe)...")

    features: List[np.ndarray] = []
    for idx, path in enumerate(X_paths, start=1):
        print(f"[{idx}/{len(X_paths)}] Procesando: {path}")
        try:
            feat = extract_pose_features(path)
            features.append(feat)
        except Exception as e:
            print(f"❌ Error con {path}: {e}")

    if not features:
        raise RuntimeError("No se pudieron extraer características de ningún vídeo.")

    X = np.stack(features)
    y_arr = np.array(y[: len(features)], dtype=int)
    return X, y_arr


def entrenar_clasificador() -> None:
    X, y = construir_dataset()

    print("🏋️ Entrenando clasificador 'técnica buena vs mala' basado en pose...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Ajuste de un modelo de covarianza para detectar poses que no se parecen
    # a las sentadillas de entrenamiento.
    print("📏 Ajustando detector de outliers en el espacio de características de pose (Mahalanobis)...")
    cov = EmpiricalCovariance()
    cov.fit(X)
    d_train = cov.mahalanobis(X)
    threshold = float(np.percentile(d_train, OUTLIER_PERCENTIL))
    print(f"   Umbral de distancia (percentil {OUTLIER_PERCENTIL}): {threshold:.3f}")

    y_pred = clf.predict(X)
    print("\n📊 Informe de clasificación (sobre el propio conjunto de entrenamiento):")
    print(classification_report(y, y_pred, target_names=["mala", "buena"]))

    os.makedirs(MODELOS_DIR, exist_ok=True)
    bundle = {
        "clf": clf,
        "cov": cov,
        "threshold": threshold,
        "outlier_percentil": OUTLIER_PERCENTIL,
    }
    joblib.dump(bundle, CLF_PATH)
    print(f"💾 Clasificador + detector de outliers guardados en: {CLF_PATH}")


def predecir_tecnica(video_path: str) -> None:
    if not os.path.exists(video_path):
        print(f"❌ Vídeo no encontrado: {video_path}")
        return

    if not os.path.exists(CLF_PATH):
        print("❌ No encuentro el clasificador entrenado.")
        print("Primero ejecuta:")
        print("  python squat_pose_technique.py --train")
        return

    print(f"🔍 Analizando técnica en (pose): {video_path}")
    bundle = joblib.load(CLF_PATH)

    # Compatibilidad hacia atrás si solo hay un LogisticRegression
    if isinstance(bundle, LogisticRegression):
        clf = bundle
        cov = None
        threshold = None
    else:
        clf = bundle.get("clf")
        cov = bundle.get("cov")
        threshold = bundle.get("threshold")

    feat = extract_pose_features(video_path)

    # 1) Comprobación de outlier en espacio de pose.
    # Ahora solo avisa, pero NO corta la predicción: siempre devuelve BUENA/MALA.
    if cov is not None and threshold is not None:
        d = float(cov.mahalanobis(feat.reshape(1, -1))[0])
        if d > threshold:
            print("\n⚠️ La pose del vídeo es muy diferente a las sentadillas usadas para entrenar.")
            print("   Probablemente no sea una sentadilla, o es una variación/cámara muy distinta.")
            print(f"   Distancia Mahalanobis: {d:.2f}  (umbral: {threshold:.2f})")

    proba = clf.predict_proba(feat.reshape(1, -1))[0]
    pred = clf.predict(feat.reshape(1, -1))[0]

    # Interpretamos la clase 0 como "BUENA" y la clase 1 como "MALA".
    # Esto corrige el comportamiento observado en el que el modelo
    # parecía invertir las etiquetas buena/mala.
    etiqueta = "BUENA" if pred == 0 else "MALA"

    # Reordenamos también las probabilidades para que coincidan
    # con esta interpretación.
    prob_buena = proba[0]  # clase 0
    prob_mala = proba[1]   # clase 1

    max_proba = float(max(prob_buena, prob_mala))
    if max_proba < 0.6:
        print("\n⚠️ El modelo no está muy seguro de la predicción (baja confianza).")
        print("   Es posible que el vídeo no muestre una sentadilla clara como las del entrenamiento.")

    print(f"\n✅ Predicción de técnica (pose): {etiqueta}")
    print(f"   Prob(mala):  {prob_mala*100:5.2f}%")
    print(f"   Prob(buena): {prob_buena*100:5.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrenar y usar un clasificador de técnica de sentadilla (bien/mal) usando pose (MediaPipe)."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entrena el clasificador usando los vídeos en videos/sentadilla/bien y videos/sentadilla/mal.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Ruta a un vídeo para evaluar la técnica (usa el clasificador ya entrenado).",
    )

    args = parser.parse_args()

    if args.train:
        entrenar_clasificador()
    elif args.predict:
        predecir_tecnica(args.predict)
    else:
        print(
            "Uso:\n"
            "  python squat_pose_technique.py --train\n"
            "    -> entrena el clasificador bien/mal con tus vídeos usando pose (MediaPipe).\n\n"
            "  python squat_pose_technique.py --predict ruta/al/video.mp4\n"
            "    -> predice si la técnica es BUENA o MALA en ese vídeo."
        )

