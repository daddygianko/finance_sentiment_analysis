import argparse
import Modelos as mo
import Preprocesamiento as pp
import os

def Main():   
    ap = argparse.ArgumentParser(prog='Programa de Descargas', 
                                 description='Programa que permite descargar de antemano los pesos de los modelos.')
    ap.add_argument('-m', '--model', type=str, default='all', choices=['all', 'no_da', 'eda', 'nlpaug'],
                    help='Seleccionar \'all\' para descargar todos los pesos o seleccionar entre \'no_da\', \'eda\', \'nlpaug\' para descargar solo los pesos elegidos. Valor por defecto \'all\'')
    ap.add_argument('-p', '--path', type=str, 
                    help='Ruta de la carpeta donde se descargarán los pesos de los modelos. Ejemplo: \'c:\\models\'' )
    args = ap.parse_args()
    
    weights_path = os.getcwd() + '\\modelos_guardados\\FinancialPhraseBank\\'
    if args.path:
        weights_path=args.path+'\\'
    
    print(f'\nDESCARGANDO LOS PESOS ...\n')
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if args.model == 'all':
        for m in ['no_da', 'eda', 'nlpaug']:
            mo.descargar_modelos(weights_path,f'pesos_Model_RoBERTa_LSTM_{m}')
            mo.descargar_modelos(weights_path,f'pesos_Model_RoBERTa_BiLSTM_{m}')
            mo.descargar_modelos(weights_path,f'pesos_Model_RoBERTa_GRU_{m}')
    else:
        mo.descargar_modelos(weights_path,f'pesos_Model_RoBERTa_LSTM_{args.model}')
        mo.descargar_modelos(weights_path,f'pesos_Model_RoBERTa_BiLSTM_{args.model}')
        mo.descargar_modelos(weights_path,f'pesos_Model_RoBERTa_GRU_{args.model}')
    
    print(f'\n############# FIN DEL PROGRAMA #############\n')
    
if __name__ == '__main__':
    Main()