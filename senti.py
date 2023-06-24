import argparse
import Modelos as mo
import Preprocesamiento as pp
import numpy as np
import os
#import tensorflow as tf
#from tensorflow.keras import Model


def Main():    
    ap = argparse.ArgumentParser(prog='Sentiment Prediction Program', 
                                 description='Determines the sentiment of the sentence provided')
    ap.add_argument('-s', '--sentence', required=True, type=str, help='Enter a sentence to be classified.')
    ap.add_argument('-p', '--path', type=str, help='pretrained saved models path. Example: c:\\models' )
    args = ap.parse_args()
    
    path = os.getcwd()+'\\modelos_guardados'
    if args.path:
        path=args.path
    
    path += '\\FinancialPhraseBank\\'
    pred_class = {0:'Negative', 1:'Neutral', 2:'Positive'}
    
    print(f'\nCARGANDO MODELOS Y PESOS\n')
    modelo_lstm = mo.cargar_modelo_json(path + 'Model_RoBERTa_LSTM_eda_config.json')
    modelo_bilstm = mo.cargar_modelo_json(path + 'Model_RoBERTa_BiLSTM_eda_config.json')
    modelo_gru = mo.cargar_modelo_json(path + 'Model_RoBERTa_GRU_eda_config.json')
    
    modelo_lstm = mo.cargar_pesos(modelo_lstm, path + 'pesos_Model_RoBERTa_LSTM_eda.h5')
    modelo_bilstm = mo.cargar_pesos(modelo_bilstm, path + 'pesos_Model_RoBERTa_BiLSTM_eda.h5')
    modelo_gru = mo.cargar_pesos(modelo_gru, path + 'pesos_Model_RoBERTa_GRU_eda.h5')
    
    print(f'\nTEXTO ORIGINAL: \n{args.sentence}')
    
    texto_limpio = pp.limpieza_listas([args.sentence])[0]
    print(f'\nTEXTO LIMPIO: \n{texto_limpio}\n')
    
    token_ids, attention_mask = mo.tokenizar([texto_limpio], 180)
    
    print(f'\nCALCULANDO LAS PREDICCIONES')
    pred1 = modelo_lstm.predict([token_ids, attention_mask])
    clase1 = pred_class[np.argmax(pred1, axis=1)[0]]
    pred2 = modelo_bilstm.predict([token_ids, attention_mask])
    clase2 = pred_class[np.argmax(pred2, axis=1)[0]]
    pred3 = modelo_gru.predict([token_ids, attention_mask])
    clase3 = pred_class[np.argmax(pred3, axis=1)[0]]
    pred4 = mo.predict_avg_ensemble(pred1, pred2, pred3)
    clase4 = pred_class[np.argmax(pred4, axis=1)[0]]
    pred5 = mo.predict_vote_ensemble(pred1, pred2, pred3)[0]
    clase5 = pred_class[pred5]
    
    print(f'Pred RoBERTa-LSTM: {clase1}')
    print(f'Pred RoBERTa-BiLSTM: {clase2}')
    print(f'Pred RoBERTa-GRU: {clase3}')
    print(f'Pred Avg Ensemble: {clase4}')
    print(f'Pred Vote Ensemble: {clase5}')

    print(f'\n############# FIN DEL PROGRAMA #############\n')
    
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    Main()
