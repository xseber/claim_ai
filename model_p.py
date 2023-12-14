# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:45:20 2023

@author: xseber
"""

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import skops.io as sio

import matplotlib.pyplot as plt

def age_gr(x):
    if x <=20 :
        return 'Child'
    elif x>20 and x<=40:
        return 'Teen'
    elif x>40 and x <= 65:
        return 'Adult'
    else:
        return 'Old'

data = pd.read_csv('database_2.csv', delimiter=',')

encode = OneHotEncoder(drop='first')
data['age_group'] = data['age'].transform(age_gr)
#data['pp'] = data['pp'].apply(lambda x: 1 if x == 'High' else 0)
#data['hp_rank'] = data['hp_rank'].apply(lambda x: 1 if x == 'Private' else 0)
#data['hp_area'] = data['hp_area'].apply(lambda x: 1 if x == 'City' else 0)
#data['icd9'] = data['icd9'].apply(lambda x: 1 if x == 'Open' else 0)

encode.fit(np.array(data[['pp','age_group', 'hp_rank', 'hp_area', 'icd9']]))
X = encode.transform(np.array(data[['pp','age_group', 'hp_rank','hp_area', 'icd9']])).toarray()
x = np.concatenate((X, np.array((data['cost']/5000)).reshape(-1,1)
                    , np.array((data['days_of_admit'])).reshape(-1,1)/10), axis=1)
y = data['Fraud?'].apply(lambda x : 1 if x == 'Yes' else 0)
model = RandomForestClassifier(n_estimators=10)

model.fit(x,y)
print(model.score(x,y))

result = model.predict(x)


tree_index = 0
plt.figure(figsize=(12, 8))
_ = plot_tree(model.estimators_[tree_index]
              , feature_names = ['pp1','pp2','ag1','ag2'
                                 ,'ag3', 'hp_rank','hp_area', 'icd9'
                                 , 'cost', 'admitdays']
              , class_names = ['no', 'yes'], filled=True, rounded=True)
plt.show()

sio.dump(model,'model.md')
sio.dump(encode, 'encode.md')


"""
import skops.io as sio
sio.dump(model,'model.md')
sio.dump(encode, 'encode.md')
model = sio.load('model.md',trusted= True)
"""
