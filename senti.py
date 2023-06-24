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
    ap.add_argument('-m', '--model', type=str, default='eda', help='Select between \"no_da\", \"eda\", or \"nlpaug\" to decide which trained models to use.')
    ap.add_argument('-p', '--path', type=str, help='pretrained saved models path. Example: c:\\models\\' )
    args = ap.parse_args()
    
    path = os.getcwd() + '\\modelos_guardados\\FinancialPhraseBank\\'
    if args.path:
        path=args.path
    
    pred_class = {0:'Negative', 1:'Neutral', 2:'Positive'}
    
    print(f'\nCARGANDO MODELOS Y PESOS\n')
    modelo_lstm = mo.cargar_modelo_json(path + f'Model_RoBERTa_LSTM_{args.model}_config.json')
    modelo_bilstm = mo.cargar_modelo_json(path + f'Model_RoBERTa_BiLSTM_{args.model}_config.json')
    modelo_gru = mo.cargar_modelo_json(path + f'Model_RoBERTa_GRU_{args.model}_config.json')
    
    mo.descargar_modelos(path,f'pesos_{modelo_lstm.name}')
    mo.descargar_modelos(path,f'pesos_{modelo_bilstm.name}')
    mo.descargar_modelos(path,f'pesos_{modelo_gru.name}')
    
    modelo_lstm = mo.cargar_pesos(modelo_lstm, path + f'pesos_{modelo_lstm.name}.h5')
    modelo_bilstm = mo.cargar_pesos(modelo_bilstm, path + f'pesos_{modelo_bilstm.name}.h5')
    modelo_gru = mo.cargar_pesos(modelo_gru, path + f'pesos_{modelo_gru.name}.h5')
    
    print(f'\nTEXTO ORIGINAL: \n{args.sentence}')
    
    texto_limpio = pp.limpieza_listas([args.sentence])[0]
    
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
