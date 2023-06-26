# finance_sentiment_analysis

Finance Sentiment Analysis es nuestro proyecto final de carrera realizado para poder predecir el sentimiento de titulares financieros. 

Se tomó como base el modelo de predicción ensamblado descrito en el paper [Sentiment Analysis With Ensemble Hybrid Deep Learning Model](https://ieeexplore.ieee.org/document/9903622) de los autores Tan, Lee, Lim y Anbananthen.

## Como Usar

Si se desea usar la aplicación de análisis de sentimiento se deberá ejecutar el archivo _senti.py_ y pasarle como parámetro la oración a analizar utilizando -s o --sentence.
Ejemplo:
```bash
python senty.py -s '<insertar oración>'
```
Los parámetros que se pueden pasar a la aplicación son:
* -s, --sentence. La oración que será analizada. 
* -p, --path. Cambia la ruta donde se encuentran/descargarán los pesos de los modelos usados por la creación de los modelos. 
* -m, --model. Elegir que modelo preentrenado será utilizado. Existen tres posibilidades: 'no_da' para el modelo sin aumento de datos, 'eda' para el modelo que utilizó tecnicas EDA para el aumento de datos, y 'nlpaug' para el modelo entrenado con técnicas de aumento de la librería nlpaug. Por defecto utiliza la opción 'eda.
La aplicación _senti-py_ descargará los archivos de los pesos de los modelos y los de la librería NLTK si es que estos no se encontraran en la carpeta de trabajo. 

Debido a que los pesos de los modelos pesan casi 500mb cada uno, se pueden descargar de antemano utilizando la aplicación _weights_dl.py_.
Ejemplo:
```bash
python weights_dl.py
```
Los parámetros que se pueden pasar a la aplicación son:
* -p, --path. Cambia la ruta donde se descargarán los pesos de los modelos usados por la creación de los modelos. 
* -m, --model. Seleccionar 'all' para descargar todos los pesos o seleccionar entre 'no_da', 'eda', 'nlpaug' para descargar solo los pesos elegidos. Valor por defecto 'all'

Si se desea reentrenar los modelos o modificar la arquitectura se deberá utilizar el archivo jupyter notebook denominado _Modelos_dataset_FinancialPhraseBank.ipynb_. El entrenamiento se realizó en Google Colab ya que no contabamos con un dispositivo con GPU para el entrenamiento.

En la primera celda se encuentran algunas variables que determinan si se ejecuta el notebook en una pc o en google colab, si se desea crear la arquitectura de los modelos y entrenarlos o si solo se desea cargar los archivos previamente guardados. 
Si se ejecuta en Google Colab se deberá agregar la carpeta datasets y los archivos _Data_Augmentation_colab.py_, _Modelos.py_ y _Preprocesamiento.py_ al entorno de trabajo, además, los archivos guardados que se generen serán guardados en el Google Drive del usuario.


## Video de Uso



## Paquetes Necesarios

Se utilizó python 3.9.16 para la creación del proyecto y será necesario tanto para el uso de la aplicación para el análisis de sentimiento como para el notebook donde se realizó el entrenamiento de los modelos.

Para el uso de la aplicación de clasificación se necesitaran las siguientes dependencias:

* pandas, scikit-learn, nltk, unidecode, matplotlib, seaborn, tensorflow, transformers, gdown

Si se desea realizar el entrenamiento de los modelos usando el jupyter notebook se necesitarán los paquetes previamente mencionados más las siguientes dependencias:

* nlpaug, torch>=1.6.0, transformers>=4.11.3, sentencepiece, sacremoses, googletrans-py>=4.0.0

Para la instalación del paquete googletrans-py se necesitará usar pip debido a que la versión de conda no es la correcta.
´´´bash
pip install googletrans-py
´´´

## Autores
* Flores Cuenca, Luis Fernando
* Mello Loayza, Gianfranco Willy

## Agradecimientos

K. L. Tan, C. P. Lee, K. M. Lim and K. S. M. Anbananthen, "Sentiment Analysis With Ensemble Hybrid Deep Learning Model," in IEEE Access, vol. 10, pp. 103694-103704, 2022, doi: 10.1109/ACCESS.2022.3210182.