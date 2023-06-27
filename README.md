# Finance Sentiment Analysis

Finance Sentiment Analysis es nuestro proyecto final de carrera realizado para poder predecir el sentimiento de titulares financieros. El proyecto se encarga de clasificar el titular financiero en tres posibles categorías: Negativo, Neutro o Positivo.

La predicción se realiza mediante el uso de 5 modelos distintos. Tres modelos híbridos: RoBERTa-LSTM, RoBERTa-BiLSTM y RoBERTa-GRU. Y por 2 modelos ensamblados, por promedio y por votación, donde se ensamblan los tres modelos previamente mencionados. 

Además se utilizaron tres enfoques para entrenar los datos. La primera es entrenar los modelos sin realizar aumento de datos, la segunda realizando aumento de datos usando técnicas EDA, y la tercera realizando aumento de datos utilizando métodos de aumento de la librería NLPAUG.

Se tomaron como base los modelos de predicción ensamblados descritos en el paper [Sentiment Analysis With Ensemble Hybrid Deep Learning Model](https://ieeexplore.ieee.org/document/9903622) de los autores Tan, Lee, Lim y Anbananthen.

Para el entrenamiento se utilizó el dataset [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) extraído de Kaggle.

Debido al desbalance de clases se optó por 2 métodos de aumento de clases.
* La primera está basada en las técnicas de aumento de datos denominadas EDA descritas en el paper [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/pdf/1901.11196v2.pdf) de los autores Wei y Zou.
* Para la segunda se utilizó la librería [NLPAUG](https://github.com/makcedward/nlpaug/tree/master). Más precisamente, se utilizaron las siguientes técnicas de aumento de datos para oraciones: _Spelling Augmenter_, _Insert word by contextual word embeddings using RoBERTa_, _Substitute word by contextual word embeddings using RoBERTa_ y _Back Translation Augmenter_.

## Como Usar

Si se desea usar la aplicación de análisis de sentimiento se deberá ejecutar el archivo _senti.py_ y pasarle como parámetro la oración a analizar utilizando -s o --sentence.
Ejemplo:
```bash
python senty.py -s '<insertar oración>'
```
Los parámetros que se pueden pasar a la aplicación son:
* -s, --sentence. La oración que será analizada. Parámetro obligatorio.
* -p, --path. Cambia la ruta donde se encuentran/descargarán los pesos de los modelos usados por la creación de los modelos. 
* -m, --model. Elegir que modelo preentrenado será utilizado. Existen tres posibilidades: 'no_da' para el modelo sin aumento de datos, 'eda' para el modelo que utilizó tecnicas EDA para el aumento de datos, y 'nlpaug' para el modelo entrenado con técnicas de aumento de la librería nlpaug. Por defecto utiliza la opción 'eda.
La aplicación _senti-py_ descargará los archivos de los pesos de los modelos y los de la librería NLTK si es que estos no se encontraran en la carpeta de trabajo. 

Debido a que los pesos de los modelos tienen un tamaño de casi 500mb cada uno, se pueden descargar de antemano utilizando la aplicación _weights_dl.py_.
Ejemplo:
```bash
python weights_dl.py
```
Los parámetros opcionales que se pueden pasar a la aplicación son:
* -p, --path. Cambia la ruta donde se descargarán los pesos de los modelos usados por la creación de los modelos. 
* -m, --model. Seleccionar 'all' para descargar todos los pesos o seleccionar entre 'no_da', 'eda', 'nlpaug' para descargar solo los pesos elegidos. Valor por defecto 'all'

Si se desea reentrenar los modelos o modificar la arquitectura se deberá utilizar el archivo jupyter notebook denominado _Modelos_dataset_FinancialPhraseBank.ipynb_. El entrenamiento se realizó en Google Colab ya que no contabamos con un dispositivo con GPU para el entrenamiento.

En la primera celda se encuentran algunas variables que determinan si se ejecuta el notebook en una pc o en google colab, si se desea crear la arquitectura de los modelos y entrenarlos o si solo se desea cargar los archivos previamente guardados. 
Si se ejecuta en Google Colab se deberá agregar la carpeta datasets y los archivos _Data_Augmentation_colab.py_, _Modelos.py_ y _Preprocesamiento.py_ al entorno de trabajo, además, los archivos guardados que se generen serán guardados en el Google Drive del usuario.


## Video de Uso

[![Video de uso](https://img.youtube.com/vi/oOTr13q1BZs/hqdefault.jpg)](https://www.youtube.com/watch?v=oOTr13q1BZs)

## Paquetes Necesarios

Se utilizó python 3.9.16 para la creación del proyecto y será necesario tanto para el uso de la aplicación para el análisis de sentimiento como para el notebook donde se realizó el entrenamiento de los modelos.

Para el uso de la aplicación de clasificación se necesitaran las siguientes dependencias:

* pandas, scikit-learn, nltk, unidecode, matplotlib, seaborn, tensorflow, transformers, gdown

Si se desea realizar el entrenamiento de los modelos usando el jupyter notebook se necesitarán los paquetes previamente mencionados más las siguientes dependencias:

* nlpaug, torch>=1.6.0, transformers>=4.11.3, sentencepiece, sacremoses, googletrans-py>=4.0.0, openpyxl

Para la instalación del paquete googletrans-py se necesitará usar pip debido a que la versión de conda no es la correcta.
```bash
pip install googletrans-py
```

En la carpeta _environment_ se encuentra el archivo _env_proy_capstone.yaml_ con el cual se puede crear el entorno para el uso del proyecto utilizando conda.

## video de Despliegue

[![Video de despliegue](https://img.youtube.com/vi/MV8Xz1q67Ls/hqdefault.jpg)](https://www.youtube.com/watch?v=MV8Xz1q67Ls)

## Autores
* Flores Cuenca, Luis Fernando. 
* Mello Loayza, Gianfranco Willy.

