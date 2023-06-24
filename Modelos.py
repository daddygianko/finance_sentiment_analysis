from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import requests


def tokenizar(data, max_length=80):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  
    input_ids = []
    attention_masks = []
    for text in data:
        encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, padding="max_length",
                                            return_attention_mask=True, return_tensors='tf')
        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])

    input_ids = np.concatenate(input_ids, axis=0)
    attention_masks = np.concatenate(attention_masks, axis=0)
    return input_ids, attention_masks


def crear_modelo(layer3, max_length=80, cat_salida=2, name=None, save_png=False):
    #capas de entradas
    input_ids_input = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_masks_input = Input(shape=(max_length,), dtype=tf.int32, name='attention_masks')
    
    #capa roberta
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    roberta_model._name="Modelo_PreEntrenado_RoBERTa"
    out = roberta_model(input_ids_input, attention_mask=attention_masks_input)[0]
    
    #capa LSTM, BiLSTM o GRU
    capa=layer3.lower()
    if(capa=="lstm"): 
        out = LSTM(256, name="Capa_LSTM")(out)
    elif(capa)=="gru": 
        out = GRU(256, name="Capa_GRU")(out)
    else:
        out = Bidirectional(LSTM(128), name="Capa_BiLSTM")(out)
    
    #capa flatten
    out = Flatten(name="Capa_Flatten")(out)
    
    #capa densa
    out = Dense(256, activation='relu', name="Capa_Densa")(out)
    
    #capa de salida/clasificación
    out = Dense(cat_salida, activation='softmax', name="Capa_Clasificacion")(out)
    
    model = Model(inputs=[input_ids_input,attention_masks_input], outputs=out, name=name)
    
    model.layers[2].trainable = True
    
    if save_png:
        cwd=os.getcwd()
        plot_model(model, to_file=cwd+f'\\imgs\\models\\{name[:-6]}_arquitecture.png', show_layer_names=True, 
                   show_layer_activations=True, show_shapes=True)
        plot_model(model, cwd+f'\\imgs\\models\\{name[:-6]}_arquitecture_compact.png', show_layer_activations=True)
    return model
    
def plotear_historia(history, save_plot=False, name='history_plot.png'):
    keys=list(history.history.keys())
    fig, ax = plt.subplots(1, 2, figsize= (12, 6))
    #summarize history for accuracy
    history.history[keys[1]].insert(0,None)
    history.history[keys[3]].insert(0,None)
    ax[0].plot(history.history[keys[1]])
    ax[0].plot(history.history[keys[3]])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel(keys[1])
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'val'], loc='upper left')
    ax[0].xaxis.get_major_locator().set_params(integer=True)
    #plt.show()
    # summarize history for loss
    history.history[keys[0]].insert(0,None)
    history.history[keys[2]].insert(0,None)
    ax[1].plot(history.history[keys[0]])
    ax[1].plot(history.history[keys[2]])
    ax[1].set_title('model loss')
    ax[1].set_ylabel(keys[0])
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'val'], loc='upper left')
    ax[1].xaxis.get_major_locator().set_params(integer=True)
    plt.show()
    if (save_plot):
        cwd=os.getcwd()
        plt.savefig(cwd+ f'\\imgs\\history\\{name}_train_history.png')

def compilar_modelo(modelo, loss, metrics, optimizer="Adam", lr=1e-3):
    if(optimizer=="Nadam"):
        optimizador=Nadam(lr)
    else:
        optimizador=Adam(lr)
    modelo.compile(loss=loss, metrics=metrics, optimizer=optimizador)
    return modelo

def entrenar_modelo(model, X, y, epochs=1, validation_data=None, validation_split=0.0, batch_size=None, early_stopping=True, 
                    early_stopping_monitor='val_loss', patience=3, plot_history=False, save_plot=False, verbose=1):
    callback=None
    if early_stopping:
        callback = EarlyStopping(monitor=early_stopping_monitor, patience=patience, restore_best_weights=True, 
                                 mode='auto')
    history = model.fit(X,y, epochs=epochs, validation_data=validation_data, validation_split=validation_split, batch_size=batch_size, 
                        callbacks=callback, verbose=verbose)
    if plot_history:
        plotear_historia(history, save_plot=save_plot, name=model.name)
    return model

def convertir_categorias(lista, n_categorias):
    cat=to_categorical(lista,n_categorias)
    return cat

def guardar_modelo(modelo, ruta:str, save_format=None):
    modelo.save(ruta, save_format=save_format)

def cargar_modelo(ruta:str):
    modelo = load_model(ruta)
    return modelo

def guardar_pesos(modelo, ruta:str, save_format=None):
    modelo.save_weights(ruta, save_format=save_format)

def cargar_pesos(modelo, ruta:str):
    modelo.load_weights(ruta)
    return modelo

def guardar_modelo_json(modelo, ruta:str):
    modelo_json = modelo.to_json()  # O bien model.to_yaml()
    with open(ruta, 'w') as json_file:
        json_file.write(modelo_json)

def cargar_modelo_json(ruta:str):
    with open(ruta, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json, {'TFRobertaModel': TFRobertaModel})
    return loaded_model

models_id={'pesos_Model_RoBERTa_LSTM_no_da':'1-RcHDQrGbthuXIcegLJC_Jws3tetrffH',
           'pesos_Model_RoBERTa_BiLSTM_no_da':'1-VAu3BCAO_FX7_ohK4e-nQuAvsv-hpIG',
           'pesos_Model_RoBERTa_GRU_no_da':'1-eJOQv9FiUOeQnZztV0m01LZKS1dayEm',
           'pesos_Model_RoBERTa_LSTM_eda':'1etsDyZ19i0hAEGjVEPEU7ql_3Il72nJQ',
           'pesos_Model_RoBERTa_BiLSTM_eda':'1yt8VQ50VcDjbyCoy-H8Sut73A31xVNFf',
           'pesos_Model_RoBERTa_GRU_eda':'1tf3tTG4JGxHlfmUqJMRodiI1K1pBK90r',
           'pesos_Model_RoBERTa_LSTM_nlpaug':'1-J2pnrdsSyhsVhCun2Th02RBOEDS6W1u',
           'pesos_Model_RoBERTa_BiLSTM_nlpaug':'1-MjNMWEio-GY5F8VE2fr6xJYMqHp2msU',
           'pesos_Model_RoBERTa_GRU_nlpaug':'1wZt36idELjmdxg-PY2ikJRXrjrfEzZMm'}

def __download_models_from_drive(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)


def descargar_modelos(save_path, model):
    ruta_guardado=save_path+model+'.h5'
    url='https://drive.google.com/uc?id='+models_id[model]
    if not os.path.exists(ruta_guardado):
        __download_models_from_drive(url, ruta_guardado)
        

def plot_confusion_matrix(y_true, y_pred, cat_values, cat_names, title:str='Matriz de Confusión'):
    cm = confusion_matrix(y_true, y_pred, labels=cat_values)
    ax = sns.heatmap(cm, fmt="d", annot=True, xticklabels=cat_names, yticklabels=cat_names)
    ax.set_xlabel('Predicciones', fontsize=14, labelpad=5)
    ax.set_ylabel('Valores Reales', fontsize=14, labelpad=5)
    ax.set_title(title, fontsize=14, pad=10)
    plt.show()
    return cm

def subplotear_cm(lista_cm, lista_titulos,cat_names, n_rows=2, n_cols=2):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7 + 5*(n_cols-2), 3 * n_rows))
    for cm, title, ax in zip(lista_cm, lista_titulos, axes.flatten()):
        sns.heatmap(cm, fmt="d", annot=True, xticklabels=cat_names, yticklabels=cat_names, ax=ax, cmap='Blues')
        ax.set_xlabel('Predicciones', fontsize=11, labelpad=5)
        ax.set_ylabel('Valores Reales', fontsize=11, labelpad=5)
        ax.set_title(title, fontsize=12, pad=11)
    plt.tight_layout()  
    plt.show()

def predict_avg_ensemble(*lista_preds):
    pred_model = [np.sum(preds, axis=0) / 3 for preds in zip(*lista_preds)]
    return pred_model
    
def predict_vote_ensemble(*list_preds):
    pred_class=[]
    for pred in list(list_preds):
        pred_class.append(np.argmax(pred, axis=1))
    pred_model = [Counter(pred for pred in predictions).most_common(1)[0][0] for predictions in zip(*pred_class)]
    return pred_model

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return [acc, prec, rec, f1]
