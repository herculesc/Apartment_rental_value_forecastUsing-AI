'''Ativação do modelo de MLP para regresão'''
import numpy as np
import pandas as pd
import keras.models
import tensorflow as tf



def load_model(path_model):
    return keras.models.load_model(path_model)
    

def normalize(entrada):
    max_value =[26732.631, 13.5, 16.0, 12.0, 1.0, 1.0, 14000.0, 19333.3326, 10830.0, 262.363, 32.0]
    normalizar = []
    for i in range(len(max_value)):
        mx = max_value[i]
        ent = entrada[i]
        cal = (ent *1.0)/ mx
        normalizar.append(cal)
    normalizar = np.array([normalizar])
    return normalizar
        

    
    
'''main'''
# Modelo
#model = load_model('my_model')


# Entrada dos dados (area, rooms....floor)
#val_entrada = [150.0, 3.0, 2.0, 2.0, 1.0, 0.0, 1181.0, 2000.0, 384.0, 26.0, 12.0]


# Normalizar entrada
#entrada = normalize(val_entrada)
#print(entrada)

# Previsão
'''predicoes = model.predict(entrada)

# Imprimir resultado
print(f"Predito 1: {str(predicoes[0])}")'''

