from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import colored
import sklearn
import feature_engine

app = Flask(__name__)

modelo_entrenado_para_productivo = joblib.load('titanic_v122022.pkl')
FEATURES=joblib.load('FEATURES.pkl')

def generatLog(message, logType):
    f = open("logsData.log", "a")
    message = message + datetime.today().strftime('%y-%m-%d%H:%M:%S')+";" '\\n'
    f.write(message)
    if(logType==10): 
        strColor = "Yellow"
    elif(logType==30):
        strColor = "red" 

    print(colored(message, strColor))
    f.close()

@app.route("/predictOne", methods=['POST'])

def predictOne():
#Código General para cualquir API
    data = request.get_json()
    dataframe=pd.json_normalize(data)
    logStr = "0x10 - INFO - JSON transformado exitosamente -"
    print(dataframe)
    generatLog(logStr,10)

#Código Particular para este API
ids = dataframe['Id']
dataframe = dataframe[FEATURES] 

#Predicción
try:
    nomr_preds = modelo_entrenado_para_productivo.predict(dataframe)
    outPredict = np.exp(nomr_preds)
    out = {}
    for index, item in enumerate(outPredict):
        out[str(ids[index])] = round(item,2)
    print(out)


except ValueError:
    logStr = "0x30 - PredictError - Se Genero un error en la prediccion -" 
    generatLog(logStr,30)



