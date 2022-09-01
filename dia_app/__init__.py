import os
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model=pickle.load(open('./dia_app/md.pkl','rb'))


def create_app():

    app = Flask(__name__)

    @app.route('/')
    def main():
        return render_template('main.html')

    @app.route('/result', methods=['POST'])
    def result():
        d1=request.form['carat']
        d2=request.form['cut']
        d3=request.form['color']
        d4=request.form['clarity']
        d5=request.form['depth']
        d6=request.form['table']
        d7=request.form['x']
        d8=request.form['y']
        d9=request.form['z']

        arr=np.array([d1,d2,d3,d4,d5,d6,d7,d8,d9])
        df_arr=pd.DataFrame([arr],columns=['carat','cut','color','clarity','depth','table','x','y','z'])

        pred=model.predict(df_arr)

        result=np.exp(pred)

        return render_template('result.html',data=result)

    return app
