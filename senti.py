import argparse
import Modelos as mo
import Preprocesamiento as pp
import numpy as np
import os


def Main():    
    ap = argparse.ArgumentParser(prog='Análisis de Sentimientos Financiero', 
                                 description='Determina el sentimiento (Negativo, Positivo o Neutro) de la oración proporcionada.')
    ap.add_argument('-s', '--sentence', required=True, type=str, help='La oración que se desea clasificar')
    ap.add_argument('-m', '--model', type=str, default='eda', choices=['no_da', 'eda', 'nlpaug'],
                    help='Seleccionar entre \'no_da\', \'eda\', o \'nlpaug\' para decidir que modelo utilizar en la clasificación. Valor por defecto \'eda\'')
    ap.add_argument('-p', '--path', type=str, 
                    help='La ruta donde se encuentran/descargarán los pesos de los modelos a utilizar. Ejemplo: \'c:\\models\'' )
    args = ap.parse_args()
    
    configs_path = os.getcwd() + '\\modelos_guardados\\FinancialPhraseBank\\'
    weights_path = os.getcwd() + '\\modelos_guardados\\FinancialPhraseBank\\'
    if args.path:
        weights_path=args.path+'\\'
    
    pred_class = {0:'Negativo', 1:'Neutro', 2:'Positivo'}
    
    print(f'\nCARGANDO LOS MODELOS\n')
    modelo_lstm = mo.cargar_modelo_json(configs_path + f'Model_RoBERTa_LSTM_{args.model}_config.json')
    modelo_bilstm = mo.cargar_modelo_json(configs_path + f'Model_RoBERTa_BiLSTM_{args.model}_config.json')
    modelo_gru = mo.cargar_modelo_json(configs_path + f'Model_RoBERTa_GRU_{args.model}_config.json')
    
    print(f'\nDESCARGANDO LOS PESOS ...\n')
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    mo.descargar_modelos(weights_path,f'pesos_{modelo_lstm.name}')
    mo.descargar_modelos(weights_path,f'pesos_{modelo_bilstm.name}')
    mo.descargar_modelos(weights_path,f'pesos_{modelo_gru.name}')
    
    modelo_lstm = mo.cargar_pesos(modelo_lstm, weights_path + f'pesos_{modelo_lstm.name}.h5')
    modelo_bilstm = mo.cargar_pesos(modelo_bilstm, weights_path + f'pesos_{modelo_bilstm.name}.h5')
    modelo_gru = mo.cargar_pesos(modelo_gru, weights_path + f'pesos_{modelo_gru.name}.h5')
    
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
    print(f'Pred Ensamblado por Promedio: {clase4}')
    print(f'Pred Ensamblado por Votación: {clase5}')

    print(f'\n############# FIN DEL PROGRAMA #############\n')
    
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    Main()
