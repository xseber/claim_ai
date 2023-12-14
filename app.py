# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:31:02 2023

@author: xseber
"""

from quart import Quart, render_template, request, redirect

import numpy as np
import pandas as pd
import skops.io as sio

model = sio.load('model.md',trusted= True)
enc =  sio.load('encode.md',trusted= True)
app = Quart(__name__)


def age_gr(x):
    if x <=20 :
        return 'Child'
    elif x>20 and x<=40:
        return 'Teen'
    elif x>40 and x <= 65:
        return 'Adult'
    else:
        return 'Old'



@app.route('/')
async def index():
    return await render_template('index.html', result = None)

@app.route('/submit', methods=['POST'])
async def submit():
    if request.method == 'POST':
        form = await request.form 

        rank =  str(form['rank'])
        hospital_rank =  str(form['hospital_rank'])
        hospital_area =  str(form['hospital_area'])
        icd9 =  str(form['icd9'])
        age = age_gr(int(form['age']))
        cost =  (float(form['cost']))/5000
        days = int(form['days'])/10


        data_input =np.array(pd.DataFrame([[rank, str(age), hospital_rank, hospital_area, icd9]]))

        print(data_input)

        x = enc.transform(data_input).toarray()
        x = np.concatenate((x, np.array(float(days)).reshape(-1,1), np.array(float(cost)).reshape(-1,1)), axis=1)
        model_result = model.predict(x)

        print('result',model_result)
        # Process the data as needed (e.g., save to a database)

        result = {'rank_pass': True if model_result[-1] == 0 else False
                , 'word' : 'Pass!' if model_result[-1] == 0 else 'Suspicious!!!!'}
        return await render_template('index.html', result = result)
    return await render_template('index.html', result = {'rank_pass':False, 'word':'incorrect method'})

if __name__ == '__main__':
    app.run(debug=True, port = 5022)
