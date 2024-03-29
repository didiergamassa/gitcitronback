from flask import Flask,request,jsonify
import pickle
import requests
import json
import pandas as pd
import numpy as np

app=Flask(__name__)

pickle_in=open("finalized_model_auc.pkl","rb")
classifier=pickle.load(pickle_in)

data=pd.read_csv('data.skill.csv').set_index('SK_ID_CURR')
@app.route("/",methods=['GET'])
def get_default():
    return jsonify({'result': "App Running..."})

@app.route("/info_row",methods=['POST'])
def info_row():
    fax = request.get_json(force=True)#response
    x=fax['trace']
    result_list=list(x.values())
    output=classifier.predict_proba([result_list])[:,1]
    prediction=round(output[0],2)
    return jsonify({'result':prediction})

if __name__ == "__main__":
    app.run(debug=True)