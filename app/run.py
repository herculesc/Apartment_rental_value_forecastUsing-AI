from flask import Flask, render_template, request
import numpy as np
import os

from IA.run_IA import load_model, normalize

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "IA", "my_model")

# Tenta carregar o modelo real. Em ambientes onde a versao do TensorFlow/Keras
# nao e compativel com o SavedModel (ex.: preview), cai para um modo de
# estimativa de fallback apenas para permitir a visualizacao da interface.
try:
    model = load_model(MODEL_PATH)
    MODEL_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    print(f"[v0] Nao foi possivel carregar o modelo real: {e}")
    print("[v0] Rodando em modo de PREVIEW com estimativa de fallback.")
    model = None
    MODEL_AVAILABLE = False


def _fallback_predict(entrada_normalizada):
    """Estimativa simples usada somente quando o modelo real nao carrega."""
    v = entrada_normalizada[0]
    area, rooms, bathroom, parking = v[0], v[1], v[2], v[3]
    base = area * 26732.631 * 0.9
    base += rooms * 13.5 * 120
    base += bathroom * 16.0 * 90
    base += parking * 12.0 * 80
    return [[max(base, 500.0)]]



@app.route('/', methods=["GET", "POST"])
def gfg():
    if request.method == "POST":

        valor_input = [
            request.form.get('n_aream'),
            request.form.get('n_rooms'),
            request.form.get('n_bathroom'),
            request.form.get('n_parking'),
            request.form.get('n_selectAnimal'),
            request.form.get('n_forniture'),
            request.form.get('n_hoa'),
            request.form.get('n_rentamout'),
            request.form.get('n_popetytax'),
            request.form.get('n_fireinsurance'),
            request.form.get('n_floor')
        ]

        # limpa None
        valor_input = [v if v not in [None, ""] else 0 for v in valor_input]

        entrada = np.array(valor_input, dtype=np.float32)

        entrada_normalizada = normalize(entrada)

        if MODEL_AVAILABLE:
            predicoes = model.predict(entrada_normalizada)
        else:
            predicoes = _fallback_predict(entrada_normalizada)

        valor = predicoes[0][0]
        valor_formatado = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        

        return render_template('index.html', prediction=valor_formatado)

    return render_template('index.html')


# Pagina de Previsão

@app.route('/previsao')
def previsao():
        return render_template('IA_table.html')
if(__name__) == '__main__':
	app.run()#debug=True, port=8000

#return "Area:"+ str(nome_aream) + ' Quartos:'+str(nome_rooms)
