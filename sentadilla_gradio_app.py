import os
from typing import Tuple, List, Optional

import numpy as np
import joblib
import gradio as gr

import squat_pose_technique as pose_mod


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos", "sentadilla")


# ---------------------------------------------------------------------------
# Utilidades de dataset
# ---------------------------------------------------------------------------

def _ensure_dataset_dirs() -> None:
    for sub in ("bien", "mal"):
        os.makedirs(os.path.join(VIDEOS_DIR, sub), exist_ok=True)


def listar_videos_dataset() -> List[Tuple[str, str]]:
    """
    Devuelve lista de (etiqueta, nombre_archivo_relativo) para mostrar en la UI.
    """
    _ensure_dataset_dirs()
    items: List[Tuple[str, str]] = []
    for etiqueta in ("bien", "mal"):
        carpeta = os.path.join(VIDEOS_DIR, etiqueta)
        for f in sorted(os.listdir(carpeta)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                rel = os.path.join("videos", "sentadilla", etiqueta, f)
                items.append((etiqueta, rel))
    return items


def guardar_video_en_dataset(temp_path: str, etiqueta: str, nombre_destino: str | None) -> str:
    """
    Mueve un vídeo subido por el usuario a la carpeta bien/mal.
    """
    if temp_path is None:
        return "❌ No se ha recibido ningún archivo."

    _ensure_dataset_dirs()

    etiqueta = "bien" if etiqueta == "bien" else "mal"
    carpeta_destino = os.path.join(VIDEOS_DIR, etiqueta)

    base_name = nombre_destino.strip() if nombre_destino else os.path.basename(temp_path)
    if not os.path.splitext(base_name)[1]:
        # Si el usuario no puso extensión, reciclamos la del archivo temporal
        _, ext = os.path.splitext(temp_path)
        base_name += ext

    dest_path = os.path.join(carpeta_destino, base_name)

    # Si ya existe, le añadimos sufijo
    i = 1
    base, ext = os.path.splitext(dest_path)
    final_path = dest_path
    while os.path.exists(final_path):
        final_path = f"{base}_{i}{ext}"
        i += 1

    os.replace(temp_path, final_path)

    rel = os.path.relpath(final_path, BASE_DIR)
    return f"✅ Vídeo guardado en el dataset como: {rel}"


def eliminar_video_dataset(ruta_relativa: str | None) -> str:
    if not ruta_relativa:
        return "❌ Selecciona un vídeo para eliminar."

    abs_path = os.path.join(BASE_DIR, ruta_relativa)
    if not os.path.exists(abs_path):
        return f"❌ El archivo ya no existe: {ruta_relativa}"

    os.remove(abs_path)
    return f"🗑️ Vídeo eliminado: {ruta_relativa}"


# ---------------------------------------------------------------------------
# Entrenamiento de modelos
# ---------------------------------------------------------------------------

def entrenar_modelo_desde_ui(modelo: str) -> str:
    """
    Entrena el modelo de pose (MediaPipe).
    (El selector de modelo se mantiene por si en un futuro se añade otro tipo,
    pero actualmente solo se usa el de pose.)
    """
    try:
        pose_mod.entrenar_clasificador()
        return f"✅ Modelo de pose entrenado y guardado en: {pose_mod.CLF_PATH}"
    except Exception as e:
        return f"❌ Error entrenando modelo de pose: {e}"


# ---------------------------------------------------------------------------
# Predicción con modelo de pose
# ---------------------------------------------------------------------------

def analizar_video_pose(video_path: str) -> Tuple[str, dict, str]:
    """
    Analiza un vídeo usando el modelo de pose (MediaPipe).
    Devuelve: (texto_veredicto, dict_probabilidades, texto_info_extra)
    """
    if not video_path:
        return "Sube o selecciona un vídeo primero.", {}, ""

    if not os.path.exists(pose_mod.CLF_PATH):
        return (
            "❌ No encuentro el modelo de pose entrenado. Pulsa 'Entrenar' con 'Pose (MediaPipe)' o entrena vía consola.",
            {},
            "",
        )

    try:
        bundle = joblib.load(pose_mod.CLF_PATH)

        # Compatibilidad hacia atrás
        if isinstance(bundle, pose_mod.LogisticRegression):  # type: ignore[attr-defined]
            clf = bundle
            cov = None
            threshold = None
        else:
            clf = bundle.get("clf")
            cov = bundle.get("cov")
            threshold = bundle.get("threshold")

        feat = pose_mod.extract_pose_features(video_path)

        info_extra_lines: List[str] = []

        # Comprobación de "rareza" (outlier) – solo informativa
        if cov is not None and threshold is not None:
            d = float(cov.mahalanobis(feat.reshape(1, -1))[0])
            if d > threshold:
                info_extra_lines.append(
                    "⚠️ La pose del vídeo es muy diferente a las sentadillas usadas para entrenar."
                )
                info_extra_lines.append(
                    "   Probablemente no sea una sentadilla, o es una variación/cámara muy distinta."
                )
                info_extra_lines.append(f"   Distancia Mahalanobis: {d:.2f}  (umbral: {threshold:.2f})")

        proba = clf.predict_proba(feat.reshape(1, -1))[0]
        pred = clf.predict(feat.reshape(1, -1))[0]

        # Mapeo consistente: clase 0 = BUENA, clase 1 = MALA
        etiqueta = "BUENA" if int(pred) == 0 else "MALA"
        prob_buena = float(proba[0])
        prob_mala = float(proba[1])

        max_proba = max(prob_buena, prob_mala)
        if max_proba < 0.6:
            info_extra_lines.append(
                "⚠️ El modelo no está muy seguro de la predicción (baja confianza)."
            )

        veredicto = f"✅ TÉCNICA CORRECTA" if etiqueta == "BUENA" else "❌ TÉCNICA INCORRECTA"
        probabilidades = {"Mala Técnica": prob_mala, "Buena Técnica": prob_buena}

        info_extra = "\n".join(info_extra_lines)
        return veredicto, probabilidades, info_extra

    except Exception as e:
        return f"❌ Error al procesar el vídeo (pose): {e}", {}, ""


def analizar_video_general(modelo: str, video: str | None, video_dataset: str | None) -> Tuple[str, dict, str]:
    """
    Wrapper para la pestaña de inferencia: permite usar un vídeo subido
    o uno ya existente en el dataset.
    """
    # Preferimos el vídeo subido; si no hay, usamos el seleccionado del dataset
    path = video or (os.path.join(BASE_DIR, video_dataset) if video_dataset else None)

    if not path:
        return "Sube o selecciona un vídeo primero.", {}, ""

    # Actualmente solo usamos el modelo de pose.
    return analizar_video_pose(path)


# ---------------------------------------------------------------------------
# Construcción de la interfaz Gradio
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    _ensure_dataset_dirs()

    with gr.Blocks(title="Coach de Sentadillas IA") as demo:
        gr.Markdown("## 🏋️‍♂️ Coach Virtual de Sentadillas\nAnaliza tu técnica, gestiona tu dataset y entrena modelos desde una sola interfaz.")

        with gr.Tab("Entrenar modelos"):
            modelo_radio = gr.Radio(
                ["Pose (MediaPipe)"],
                value="Pose (MediaPipe)",
                label="Modelo a entrenar",
            )
            boton_entrenar = gr.Button("Entrenar modelo seleccionado")
            salida_entrenamiento = gr.Textbox(label="Log de entrenamiento")

            boton_entrenar.click(
                fn=entrenar_modelo_desde_ui,
                inputs=modelo_radio,
                outputs=salida_entrenamiento,
            )

        with gr.Tab("Probar vídeos"):
            with gr.Row():
                modelo_pred_radio = gr.Radio(
                    ["Pose (MediaPipe)"],
                    value="Pose (MediaPipe)",
                    label="Modelo a usar",
                )

            with gr.Row():
                video_input = gr.Video(label="Sube un vídeo (2–5 segundos)", sources=["upload"])

                # Selector de vídeos ya presentes en el dataset
                with gr.Column():
                    gr.Markdown("### Vídeos del dataset")
                    listado_videos = gr.Dropdown(
                        label="Selecciona vídeo existente",
                        choices=[],
                        value=None,
                    )
                    boton_refrescar_lista = gr.Button("Refrescar lista de vídeos")
                    video_preview = gr.Video(label="Vista previa", interactive=False)

            veredicto = gr.Textbox(label="Veredicto del Coach")
            probabilidades = gr.Label(label="Nivel de seguridad de la IA")
            info_extra = gr.Textbox(label="Información adicional", lines=4)
            boton_analizar = gr.Button("Analizar vídeo")

            def _refrescar_lista_videos():
                items = listar_videos_dataset()
                # Mostramos ruta relativa como valor
                choices = [ruta for _, ruta in items]
                # Actualizamos las opciones del desplegable y limpiamos la selección
                return gr.update(choices=choices, value=None)

            def _preview_video(ruta_rel: str | None) -> Optional[str]:
                if not ruta_rel:
                    return None
                return os.path.join(BASE_DIR, ruta_rel)

            boton_refrescar_lista.click(
                fn=_refrescar_lista_videos,
                inputs=None,
                outputs=listado_videos,
            )
            listado_videos.change(
                fn=_preview_video,
                inputs=listado_videos,
                outputs=video_preview,
            )

            boton_analizar.click(
                fn=analizar_video_general,
                inputs=[modelo_pred_radio, video_input, listado_videos],
                outputs=[veredicto, probabilidades, info_extra],
            )

        with gr.Tab("Gestionar dataset"):
            gr.Markdown("### Añadir vídeo al dataset")
            with gr.Row():
                video_nuevo = gr.Video(label="Sube un vídeo nuevo", sources=["upload"])
                with gr.Column():
                    etiqueta_nueva = gr.Radio(
                        ["bien", "mal"], value="bien", label="Etiqueta para este vídeo"
                    )
                    nombre_destino = gr.Textbox(
                        label="Nombre de archivo (opcional, sin ruta)", placeholder="ej. sentadilla_001.mp4"
                    )
                    boton_guardar = gr.Button("Guardar en dataset")
                    salida_guardar = gr.Textbox(label="Resultado guardado")

            def _guardar_wrapper(video_path: str | None, etiqueta: str, nombre: str) -> str:
                return guardar_video_en_dataset(video_path, etiqueta, nombre)

            boton_guardar.click(
                fn=_guardar_wrapper,
                inputs=[video_nuevo, etiqueta_nueva, nombre_destino],
                outputs=salida_guardar,
            )

            gr.Markdown("### Eliminar vídeo del dataset")
            listado_videos_gestion = gr.Dropdown(
                label="Selecciona vídeo para eliminar",
                choices=[],
                value=None,
            )
            boton_refrescar_gestion = gr.Button("Refrescar lista")
            boton_eliminar = gr.Button("Eliminar vídeo seleccionado")
            salida_eliminar = gr.Textbox(label="Resultado eliminación")

            def _refrescar_lista_gestion():
                choices = [ruta for _, ruta in listar_videos_dataset()]
                return gr.update(choices=choices, value=None)

            boton_refrescar_gestion.click(
                fn=_refrescar_lista_gestion,
                inputs=None,
                outputs=listado_videos_gestion,
            )
            boton_eliminar.click(
                fn=eliminar_video_dataset,
                inputs=listado_videos_gestion,
                outputs=salida_eliminar,
            )

        return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()

