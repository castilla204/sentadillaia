# Coach de Sentadillas IA (Análisis de Técnica con Pose)

## Integrantes

- Diego Castilla Abella
- Javier Ruiz Marraco

## 1. Objetivo de la Tarea

Nuestro proyecto cumple el objetivo de la tarea:

1. **Técnico**: Tomamos un modelo de visión (MediaPipe Pose) y entrenamos un clasificador propio (Logistic Regression) sobre nuestros vídeos, todo ejecutándose **en local**, sin APIs externas. Además, lo envolvemos en una app con interfaz web (Gradio).
2. **Pedagógico**: Dejamos un repositorio reproducible con instrucciones paso a paso, de forma que cualquier compañero pueda clonar el repo, instalar dependencias, entrenar el modelo con sus propios vídeos y evaluar su técnica de sentadilla.

## 2. El Desafío: "Especializa un Asistente"

Nuestro asistente especializado es un **Coach Virtual de Sentadillas**:

- **Dominio de conocimiento**: técnica de sentadilla (profundidad, posición del tronco, control de la rodilla, etc.).
- **Lo que hace la app**: dado un vídeo corto de una persona haciendo una sentadilla, la aplicación:
  - Detecta la pose con MediaPipe Pose.
  - Extrae ángulos relevantes (rodilla, cadera y tronco).
  - Usa un clasificador entrenado con nuestros vídeos (etiquetados como BUENA/MALA) para predecir la calidad de la técnica.
  - Devuelve un veredicto de técnica correcta o incorrecta, junto con probabilidades.

El modelo base genérico (MediaPipe Pose sin nuestro clasificador) **no sabe** si una sentadilla es buena o mala; solo da puntos de pose. Nuestro sistema SÍ lo sabe porque hemos entrenado un clasificador encima con nuestros datos.

## 3. Requisitos Técnicos Obligatorios

### 3.1 Modelo Local

- Todo se ejecuta **en local**:
  - Detección de pose con MediaPipe.
  - Cálculo de ángulos y estadísticas.
  - Clasificador `LogisticRegression` de `scikit-learn`.
- No usamos ninguna API externa (OpenAI, etc.). Solo instalamos librerías de Python.

### 3.2 Técnica de Adaptación

Hemos usado una aproximación de **ajuste sobre un modelo de base**:

- **Modelo base**: MediaPipe Pose (red neuronal pre-entrenada para detección de pose humana).
- **Adaptación**:
  - Extraemos de cada vídeo:
    - Ángulo de rodilla izquierda y derecha (medio, mínimo, máximo, desviación).
    - Ángulo de cadera izquierda y derecha.
    - Ángulo de inclinación del tronco.
  - Esto produce un vector de características por vídeo (estadísticas de esos ángulos).
  - Sobre esos vectores entrenamos un **clasificador Logistic Regression** binario:
    - Clase 0 → “BUENA” técnica.
    - Clase 1 → “MALA” técnica.
- **Justificación**:
  - Es ligero y entrenable en un portátil.
  - Aprovecha un modelo profundo pre-entrenado (MediaPipe) sin tener que hacer fine-tuning pesado.
  - Las features (ángulos) son interpretables y tienen sentido biomecánico.

### 3.3 Aplicación Funcional (Interfaz)

Hemos elegido una **interfaz web con Gradio**:

- Permite:
  - Entrenar el modelo de pose desde la propia UI.
  - Subir vídeos nuevos y etiquetarlos como BUENA/MALA.
  - Ver los vídeos que ya están en el dataset.
  - Eliminar vídeos del dataset.
  - Probar un vídeo (subido o del dataset) y ver el veredicto del coach.

Archivo principal de la app: `sentadilla_gradio_app.py`.

## 4. El Entregable: "Kit de Reproducibilidad"

Para reproducir nuestro proyecto, cualquier persona puede:

1. **Clonar el repositorio**.
2. **Instalar dependencias** con `requirements.txt`.
3. **Organizar sus propios vídeos** en las carpetas esperadas (`videos/sentadilla/bien` y `videos/sentadilla/mal`).
4. **Entrenar el modelo** desde la interfaz Gradio.
5. **Probar vídeos** y ver si la técnica se clasifica como BUENA o MALA.

## 5. Estructura del Repositorio

Nuestro repositorio real es:

```text
sentadilla-ia/
│
├── modelos/                      # Clasificadores entrenados (joblib)
│   └── sentadilla_pose_bien_mal.joblib
│
├── videos/
│   └── sentadilla/
│       ├── bien/                 # Vídeos etiquetados como técnica buena
│       └── mal/                  # Vídeos etiquetados como técnica mala
│
├── sentadilla_gradio_app.py      # App principal de Gradio (interfaz)
├── squat_pose_technique.py       # Lógica de entrenamiento y predicción en consola
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este archivo
```

> Nota: seguimos la idea de separar datos (`videos/`), modelos (`modelos/`) y código (`*.py`).  
> Podríamos migrar `sentadilla_gradio_app.py` y `squat_pose_technique.py` a una carpeta `src/`, pero para este curso hemos optado por mantenerlo plano por simplicidad.

## 6. README como Guía Paso a Paso

### 6.1 Título

**Coach Virtual de Sentadillas con IA (MediaPipe + Logistic Regression)**

### 6.2 Integrantes

- Diego Castilla Abella
- Javier Ruiz Marraco

### 6.3 Descripción

La app evalúa la técnica de una sentadilla a partir de un vídeo corto:

- Detecta la pose humana con MediaPipe.
- Calcula ángulos de rodilla, cadera y tronco a lo largo del vídeo.
- Resume esos ángulos en estadísticas (media, min, max, desviación).
- Con un modelo entrenado por nosotros (buena/mala) clasifica la sentadilla.

Resuelve el problema de **dar feedback rápido y objetivo** sobre la calidad de la sentadilla sin necesidad de un entrenador presente.

### 6.4 Modelo Base

- **Modelo**: `MediaPipe Pose` (versión clásica 0.10.14).
- **Por qué**:
  - Robusto para detectar landmarks del cuerpo humano.
  - Funciona en CPU y en tiempo real.
  - Nos da justo lo que necesitamos: coordenadas de cadera, rodilla, tobillo y hombros.

Encima de esto, usamos un **clasificador Logistic Regression** de `scikit-learn` como “capa de decisión” entrenada con nuestros vídeos etiquetados.

### 6.5 Técnica de Adaptación

- **Tipo**: ajuste de un clasificador sobre características extraídas de un modelo de visión.
- **Pipeline**:
  1. De cada frame del vídeo, obtenemos landmarks (puntos del cuerpo).
  2. Calculamos:
     - Ángulo de rodilla izquierda y derecha (cadera-rodilla-tobillo).
     - Ángulo de cadera izquierda y derecha (rodilla-cadera-hombro).
     - Ángulo del tronco respecto a la vertical.
  3. Por cada serie de ángulos a lo largo del vídeo calculamos:
     - Media, mínimo, máximo y desviación típica.
  4. Obtenemos un vector numérico por vídeo y entrenamos un `LogisticRegression`:
     - 0 → BUENA.
     - 1 → MALA.

Además, entrenamos un modelo de covarianza (`EmpiricalCovariance`) para detectar vídeos **muy diferentes** a nuestras sentadillas de entrenamiento (outliers), y mostramos un aviso en esos casos.

### 6.6 Dataset

- **Origen**: nuestros propios vídeos de sentadilla, recortados a clips cortos (2–5 segundos).
- **Estructura**:
  - `videos/sentadilla/bien` → vídeos con técnica correcta.
  - `videos/sentadilla/mal` → vídeos con técnica incorrecta.
- **Tamaño actual** (ejemplo, ajustadlo a la realidad):
  - 30 vídeos en `bien`.
  - 30 vídeos en `mal`.
- **Preprocesamiento**:
  - Todos los vídeos se leen con OpenCV.
  - Se procesan máximo 200 frames por vídeo.
  - Se descartan frames donde MediaPipe no detecta pose.

### 6.7 Instrucciones de Instalación y Ejecución

#### Clonado e instalación

```bash
git clone [url-de-vuestro-repositorio]
cd sentadilla-ia

# Crear entorno virtual (opcional, pero recomendado)
python -m venv .venv
.\.venv\Scripts\activate    # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

#### Organización de vídeos

Colocar los vídeos en:

```text
videos/sentadilla/bien/*.mp4   # técnica buena
videos/sentadilla/mal/*.mp4    # técnica mala
```

#### Entrenar desde consola (opcional)

```bash
python squat_pose_technique.py --train
```

Esto:

- Busca vídeos en `videos/sentadilla/bien` y `videos/sentadilla/mal`.
- Extrae características de pose.
- Entrena el clasificador.
- Guarda el modelo en `modelos/sentadilla_pose_bien_mal.joblib`.

#### Lanzar la app Gradio (recomendado)

```bash
python sentadilla_gradio_app.py
```

- Se abrirá (o te dará una URL tipo `http://127.0.0.1:7860`) con tres pestañas:
  - **Entrenar modelos**: entrena el modelo de pose desde la interfaz.
  - **Probar vídeos**: sube un vídeo o selecciona uno existente y ve el veredicto.
  - **Gestionar dataset**: añadir nuevos vídeos etiquetados y eliminar los que no quieras.

### 6.8 Ejemplos de Uso

1. **Vídeo con técnica buena**:

   - Acción: subimos `buenatecnica.mp4`.
   - Resultado típico (ejemplo):

   > Técnica correcta  
   > Prob(mala):  5.12%  
   > Prob(buena): 94.88%

2. **Vídeo con técnica mala**:

   - Acción: subimos `malatecnica.mp4`.
   - Resultado (ejemplo):

   > Técnica incorrecta  
   > Prob(mala):  97.43%  
   > Prob(buena): 2.57%

3. **Vídeo raro (no sentadilla o plano muy distinto)**:

   - Acción: probamos un vídeo con varios ángulos a la vez o pedaleando.
   - Resultado (ejemplo):

   > La pose del vídeo es muy diferente a las sentadillas usadas para entrenar.  
   > … (aviso de outlier)  
   > Predicción igualmente mostrada, pero con advertencia de baja confianza.

### 6.9 Referencias

- MediaPipe Pose: documentación oficial.
- Scikit-learn: `LogisticRegression`, `EmpiricalCovariance`.
- Gradio: documentación de interfaces web rápidas para ML.

### 6.10 Autoevaluación

- **Lo más difícil**:
  - Conseguir un modelo razonablemente estable con pocos vídeos y variaciones de cámara.
  - Ajustar las etiquetas BUENA/MALA y el detector de outliers para que no bloquee todos los vídeos.
- **Resultados**:
  - El modelo diferencia razonablemente bien entre sentadillas claramente buenas y claramente malas en nuestro dataset.
  - Detecta algunos vídeos “raros” y lanza avisos.
- **Mejoras futuras**:
  - Añadir feedback más detallado (por ejemplo: “te falta profundidad”, “tronco muy inclinado”, etc.).
  - Probar un modelo más avanzado de pose o combinar con un modelo de vídeo.
  - Añadir almacenamiento de históricos de usuarios.

