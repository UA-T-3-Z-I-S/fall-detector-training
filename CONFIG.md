# CONFIGURACIONES

Desarrollo y Entorno Virtual

Versión de **Python - 3.11.9**

[Link de Descarga Automática](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe "Py 3.11.9")

Verificar Versión **python --version**

### RUTAS DEL DATASET


#### VIDEOS ORIGINALES (SIN PROCESAR)

##### ENTRENAMIENTO por ETAPAS

DATASET_TRAIN_CAIDA_1=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/train/caida_1

DATASET_TRAIN_NO_CAIDA_1=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/train/no_caida_1

DATASET_TRAIN_CAIDA_2=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/train/caida_2

DATASET_TRAIN_NO_CAIDA_2=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/train/no_caida_2

DATASET_TRAIN_CAIDA_3=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/train/caida_3

DATASET_TRAIN_NO_CAIDA_3=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/train/no_caida_3


##### VALIDACIÓN

DATASET_VAL_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/val/caida

DATASET_VAL_NO_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/val/no_caida


##### TEST

DATASET_TEST_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/test/caida

DATASET_TEST_NO_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/videos/test/no_caida


#### BUFFERS PROCESADOS (.NPY)

##### ENTRENAMIENTO por ETAPAS

BUFFER_TRAIN_CAIDA_1=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/train/caida_1

BUFFER_TRAIN_NO_CAIDA_1=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/train/no_caida_1

BUFFER_TRAIN_CAIDA_2=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/train/caida_2

BUFFER_TRAIN_NO_CAIDA_2=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/train/no_caida_2

BUFFER_TRAIN_CAIDA_3=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/train/caida_3

BUFFER_TRAIN_NO_CAIDA_3=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/train/no_caida_3


##### VALIDACIÓN

BUFFER_VAL_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/val/caida

BUFFER_VAL_NO_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/val/no_caida


##### TEST

BUFFER_TEST_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/test/caida

BUFFER_TEST_NO_CAIDA=C:/Users/sebas/OneDrive/Desktop/SISTEMA DE CAIDAS/datasets/procesados/test/no_caida

### ENTORNO DE DESARROLLO

##### REQUIREMENTS.TXT

Crear y actualizar el archivo **pip freeze > requirements.txt**

Instalar todas las librerias **pip install -r requirements.txt**

##### ENTORNO VIRTUAL

Crear entorno virtual **py -3.11 -m venv venv**

Activar entorno virtual **venv\Scripts\activate**

Desactivar entorno virtual **deactivate**
