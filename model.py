# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:45:20 2023

@author: xseber
"""

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import skops.io as sio

data = pd.read_csv('database.csv', delimiter=';')

encode = OneHotEncoder(drop='first')
encode.fit(np.array(data[['sickness', 'rank_occupation'
                                  , 'hospital_rank', 'age' ]]))

x = encode.transform(np.array(data[['sickness', 'rank_occupation'
                                  , 'hospital_rank', 'age' ]])).toarray()

model = OneClassSVM(kernel = 'sigmoid')

#X = np.concatenate((x, np.array(np.log(np.log(np.log(data['cost'])))).reshape(-1,1)), axis=1)
X = x
model.fit(np.array(X))

res = model.predict(X)
sio.dump(model,'model.md')
sio.dump(encode, 'encode.md')


"""
import skops.io as sio

model = sio.load('model.md',trusted= True)
"""
