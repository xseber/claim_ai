# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:45:20 2023

@author: xseber
"""

from sklearn.svm import OneClassSVM

import pandas as pd
import numpy as np
import skops.io as sio

data = pd.read_csv('database.csv', delimiter=';')



model = OneClassSVM(kernel = 'rbf')

model.fit(np.array(data))


#obj = sio.dump(model,'model.md')

"""
import skops.io as sio

model = sio.load('model.md',trusted= True)
"""
