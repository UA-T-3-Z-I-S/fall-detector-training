# fall-detector-training

Entrenamiento del Modelo IA CNN+LSTM

Modelo base Seleccioando **EfficienteNetB0** *(Reconoce formas humanas, fondos, luz, etc)*

DataSet seleccionados

Limpieza de los DataSet *(vídeos no utiles)*

Procesamento de los Videos

* Redimencionamiento
* Normalización

Salida Vectores *(números)*

Entrenamiento del LSTM *(desde 0)*

Evaluación del Modelo *(Dataset no visto en el entrenamiento)*


# Graph TD

A [Lectura de video con OpenCV] --> B[Generación de buffer: 12/16/32 frames]
B --> C[Preprocesamiento por frame: Redimensionar (224x224) + Normalizar]
C --> D[Extracción de vectores con EfficientNetB0 (CNN)]
D --> E[Entrada secuencial al LSTM]
E --> F[Predicción: Caída o No Caída]
F --> G[Alarma en tiempo real si se detecta caída]


# Graph TD

A[Lectura de buffer con OpenCV] --> B[Preprocesamiento: Resize + Normalize]
B --> C[EfficientNetB0 fine-tuneado]
C --> D[Vectores por frame]
D --> E[LSTM]
E --> F[Predicción: Caída o No Caída]
F --> G[Alarma en tiempo real]
