from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import keras.models
import tensorflow as tf
from IA.run_IA import load_model, normalize

model = load_model('IA/my_model')
app = Flask(__name__)

'''
@app.route('/')
def index_view():
	return render_template('index.html')'''

@app.route('/', methods =["GET", "POST"])
def gfg():
        if request.method == "POST":
                nome_aream = request.form.get('n_aream')
                nome_rooms = request.form.get('n_rooms')
                nome_batroom = request.form.get('n_bathroom')
                nome_parking  = request.form.get('n_parking')
                selectAnimal = request.form.get('n_selectAnimal')
                selectForniture = request.form.get('n_forniture')
                nome_hoa  = request.form.get('n_hoa')
                nome_rentamout = request.form.get('n_rentamout')
                nome_popetytax = request.form.get('n_popetytax')
                nome_fireinsurance = request.form.get('n_fireinsurance')
                nome_floor = request.form.get('n_floor')
                
                # Entrada = []
                valor_input = [nome_aream, nome_rooms, nome_batroom, nome_parking, selectAnimal, selectForniture, nome_hoa, nome_rentamout, nome_popetytax, nome_fireinsurance, nome_floor]
                entrada = list(np.float_(valor_input))

                # Normalização da entrda
                entrada_normalizada = normalize(entrada)
                predicoes = model.predict(entrada_normalizada)

                str_previcoes = str(predicoes[0])
                remov_back = str_previcoes[1:-1]
                
                

                # Return 'Previsão: {}'.format(str(predicoes[0]))
                return 'Previsão: {}'.format(remov_back)

        return render_template('index.html')

# Pagina de Previsão

@app.route('/previsao')
def previsao():
        return rende_template('IA_table.html')
if(__name__) == '__main__':
	app.run()#debug=True, port=8000

#return "Area:"+ str(nome_aream) + ' Quartos:'+str(nome_rooms)
