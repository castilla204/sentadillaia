# Coach Virtual de Sentadillas con IA (Análisis de Técnica por Video)

## Integrantes
Diego Castilla Abella  
Javier Ruiz Marraco  

## Descripción del proyecto
Esta aplicación actúa como un entrenador personal virtual capaz de analizar vídeos cortos de una persona realizando una sentadilla y determinar si la técnica es correcta o incorrecta. El sistema resuelve el problema de la falta de supervisión en los entrenamientos en casa, previniendo lesiones mediante la evaluación automática de la calidad del movimiento. A partir de un vídeo de pocos segundos, la aplicación procesa la secuencia, extrae información biomecánica clave y emite un veredicto fundamentado en un modelo entrenado con ejemplos reales.

## Modelo base y exploración inicial
Durante la fase inicial del proyecto se contempló el uso del modelo **MCG-NJU/videomae-base-finetuned-kinetics** de Hugging Face, un transformer de vídeo pre-entrenado para reconocimiento de acciones. Sin embargo, tras realizar las primeras pruebas, los resultados obtenidos no fueron satisfactorios y se decidió descartar este enfoque por varias razones.

El modelo VideoMAE está diseñado para tareas de clasificación de acciones genéricas (correr, saltar, nadar, etc.) y no para evaluar la calidad técnica de un movimiento tan específico como la sentadilla. Al emplearlo como extractor de características y entrenar un clasificador ligero sobre sus representaciones, se observó que las predicciones eran poco fiables y no lograban diferenciar correctamente entre técnica buena y mala, especialmente con el reducido número de vídeos disponibles (decenas de ejemplos). El modelo requería una cantidad mucho mayor de datos para ajustarse al dominio concreto, y el fine‑tuning completo era inviable por la falta de recursos computacionales.

Además, el coste computacional de VideoMAE es elevado: al ser un transformer de vídeo, necesita una GPU para procesar los clips de forma ágil, lo que dificultaba su ejecución en ordenadores portátiles convencionales y rompía el requisito de que todo funcionase 100% en local. La extracción de características con este modelo resultaba lenta y consumía mucha memoria, haciendo la experiencia de usuario poco práctica.

Por el contrario, el enfoque basado en **MediaPipe Pose** demostró ser mucho más eficaz para este problema. MediaPipe, desarrollado por Google, es una red neuronal profunda (basada en BlazePose) entrenada específicamente para detectar landmarks corporales a partir de imágenes. Se eligió este modelo por varias ventajas: es robusto, funciona en CPU en tiempo real, ofrece directamente los puntos anatómicos necesarios (hombros, caderas, rodillas, tobillos) y, al ser un modelo especializado en pose, permite extraer características interpretables con sentido biomecánico.

## Técnica de adaptación
El enfoque de adaptación consiste en construir un **clasificador estadístico sobre características derivadas de la pose**. Para cada vídeo se procesan los fotogramas con MediaPipe, obteniendo las coordenadas de los landmarks. A partir de ellas se calculan tres ángulos fundamentales a lo largo de la secuencia:

- Ángulo de rodilla (cadera‑rodilla‑tobillo) para ambas piernas.
- Ángulo de cadera (rodilla‑cadera‑hombro) para ambas piernas.
- Ángulo de inclinación del tronco respecto a la vertical.

Para cada serie temporal de ángulos se extraen cuatro estadísticos: media, mínimo, máximo y desviación típica. El resultado es un vector de características por vídeo que resume el movimiento. Sobre estos vectores se entrena un modelo de regresión logística (`LogisticRegression` de scikit‑learn) con dos clases: "técnica buena" (0) y "técnica mala" (1). Este clasificador constituye la capa de decisión final que, dada una nueva grabación, predice la calidad de la sentadilla.

Además, se incorpora un detector de novedad basado en una estimación de covarianza empírica (`EmpiricalCovariance`) para identificar vídeos cuyas características difieren significativamente del conjunto de entrenamiento, mostrando una advertencia cuando la predicción pueda ser poco fiable.

Este esquema permite aprovechar un modelo profundo pre‑entrenado (MediaPipe) sin necesidad de realizar un fine‑tuning costoso, y al mismo tiempo ofrece un sistema interpretable y entrenable con pocos ejemplos. Los resultados obtenidos son notoriamente mejores que con VideoMAE: el sistema alcanza una precisión aceptable en la clasificación, es explicable (se puede saber por qué una sentadilla se considera buena o mala) y se ejecuta sin problemas en CPU.

## Dataset
Dado que el modelo base no distingue entre una sentadilla bien o mal ejecutada, se creó un conjunto de datos propio a partir de grabaciones reales. Los vídeos, de entre dos y cinco segundos de duración y con una única repetición, se recortaron manualmente y se etiquetaron como "bien" o "mal" atendiendo a criterios biomecánicos (profundidad suficiente, control de rodillas, posición del tronco, etc.).

La estructura de carpetas refleja directamente estas etiquetas:

- `videos/sentadilla/bien/` contiene los ejemplares de técnica correcta.
- `videos/sentadilla/mal/` contiene los de técnica incorrecta.

Actualmente el dataset cuenta con aproximadamente 30 vídeos por categoría, aunque el sistema está diseñado para que cualquier usuario pueda ampliarlo fácilmente añadiendo nuevos vídeos a las carpetas correspondientes.

## Instrucciones de instalación y ejecución
Para reproducir el proyecto en un entorno local, se deben seguir los siguientes pasos:

1. Clonar el repositorio desde la URL proporcionada y acceder al directorio creado.
2. Crear y activar un entorno virtual de Python (opcional pero recomendado).
3. Instalar las dependencias necesarias ejecutando `pip install -r requirements.txt`.
4. Colocar los vídeos propios en las carpetas `videos/sentadilla/bien/` y `videos/sentadilla/mal/` respetando los formatos habituales (MP4, AVI, etc.).
5. Para entrenar el clasificador desde la línea de comandos, ejecutar `python squat_pose_technique.py --train`.  
   Este comando procesa todos los vídeos, extrae las características de pose, entrena el modelo y lo guarda en la carpeta `modelos/` con el nombre `sentadilla_pose_bien_mal.joblib`.
6. Lanzar la interfaz gráfica con `python sentadilla_gradio_app.py`.  
   Esto abrirá una aplicación web local (por defecto en `http://127.0.0.1:7860`) con tres pestañas:
   - **Entrenar modelos**: permite reentrenar el clasificador desde la interfaz.
   - **Probar vídeos**: admite la subida de un vídeo nuevo o la selección de uno existente en el dataset para obtener la predicción.
   - **Gestionar dataset**: facilita la incorporación de nuevos vídeos etiquetados y la eliminación de aquellos que ya no sean deseados.

## Ejemplos de uso
- Al subir un vídeo con una sentadilla técnicamente correcta, la aplicación devuelve un mensaje similar a:  
  "Técnica correcta. Probabilidad de técnica mala: 5.12%, probabilidad de técnica buena: 94.88%".
- Para un vídeo con una ejecución claramente incorrecta (por ejemplo, rodillas valgas o falta de profundidad), el resultado podría ser:  
  "Técnica incorrecta. Probabilidad de técnica mala: 97.43%, probabilidad de técnica buena: 2.57%".
- Si se prueba un vídeo que no corresponde a una sentadilla (por ejemplo, una persona pedaleando o con planos muy diferentes), el detector de novedad muestra una advertencia:  
  "La pose del vídeo es muy diferente a las sentadillas usadas para entrenar. La predicción puede ser poco fiable", seguida de la clasificación obtenida.

## Referencias
- MediaPipe Pose: documentación oficial de Google.
- Scikit‑learn: documentación de los módulos `LogisticRegression` y `EmpiricalCovariance`.
- Gradio: documentación para la creación de interfaces web de machine learning.

## Autoevaluación
El principal desafío ha sido lograr un modelo razonablemente estable con un número limitado de vídeos y con cierta variabilidad en las condiciones de grabación (iluminación, ángulo de cámara, vestimenta). Ajustar el umbral del detector de outliers para que no descartara demasiados vídeos válidos también requirió varias iteraciones.

La decisión de descartar Hugging Face y optar por MediaPipe fue clave: los transformers de vídeo como VideoMAE, aunque potentes para tareas genéricas, no se adaptaban bien a nuestro problema específico con pocos datos y requerían recursos computacionales que dificultaban la ejecución local. En cambio, MediaPipe proporcionó una base sólida y eficiente.

En conjunto, el sistema diferencia adecuadamente entre sentadillas claramente buenas y claramente malas dentro del dominio de los ejemplos de entrenamiento, y emite advertencias cuando se enfrenta a entradas atípicas. Como líneas de mejora futura se plantea:

- Añadir retroalimentación más detallada (por ejemplo, "te falta profundidad" o "tronco demasiado inclinado").
- Probar modelos de pose más avanzados o combinar la información de ángulos con representaciones profundas.
- Incorporar un histórico de usuarios para realizar un seguimiento de la evolución de la técnica.
