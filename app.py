# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:31:02 2023

@author: xseber
"""

from quart import Quart, render_template, request, redirect

import numpy as np

import skops.io as sio

model = sio.load('model.md',trusted= True)

app = Quart(__name__)

@app.route('/')
async def index():
    return await render_template('index.html', result = None)

@app.route('/submit', methods=['POST'])
async def submit():
    
    if request.method == 'POST':
        form = await request.form 
        rank =  form['rank'][-1]
        sickness =  form['sickness'][-1]
        hospital_rank =  form['hospital_rank'][-1]
        age = form['age'][-1]
        cost =  form['cost']

        data_input =np.array([[int(rank), int(sickness), int(hospital_rank), int(age), float(cost)]])

        model_result = model.predict(data_input)[-1]
        print(data_input)
        print('result',model_result)
        # Process the data as needed (e.g., save to a database)

        result = {'rank_pass': True if model_result == 1 else False
                , 'word' : 'Pass' if model_result == 1 else 'Anomaly!!'}
        return await render_template('index.html', result = result)
    return await render_template('index.html', result = {'rank_pass':False, 'word':'incorrect method'})

if __name__ == '__main__':
    app.run(debug=True, port = 5022)
