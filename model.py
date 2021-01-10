import lightgbm as lgb
import numpy as np
import pickle
#Python3
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("demo_dataset.csv",index_col=0)
X,y=df.drop('classification',axis=1),df['classification']

cfg={'random_state': 42, 'objective': 'multiclass', 'num_leaves': 31, 'num_iterations': 400, 'num_classes': 4, 'max_bin': 510, 'learning_rate': 0.05, 'class_weight': {0: 0.4763770180436847, 1: 0.7802706552706553, 2: 0.8719135802469136, 3: 0.8714387464387464}}
clf = lgb.LGBMClassifier(**cfg)


#Fitting model with trainig data
clf.fit(X, y)

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))