import sqlite3
import pandas as pd
import numpy as np
import pickle
from category_encoders import OrdinalEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score

conn = sqlite3.connect('pj3.db')

cur = conn.cursor()

cur.execute("SELECT * FROM diamonds")

dia_data=cur.fetchall()

df=pd.DataFrame(data=dia_data,columns=['Unnamed: 0','carat','cut','color','clarity','depth','table','price','x','y','z'])

df=df.drop('Unnamed: 0',axis=1).drop_duplicates(ignore_index=True)

dfc=df.copy()

target ='price'

iqr_target=dfc[target].quantile(0.75)-dfc[target].quantile(0.25)

iqr_carat=dfc.carat.quantile(0.75)-dfc.carat.quantile(0.25)
iqr_depth=dfc.depth.quantile(0.75)-dfc.depth.quantile(0.25)
iqr_table=dfc.table.quantile(0.75)-dfc.table.quantile(0.25)
iqr_x=dfc.x.quantile(0.75)-dfc.x.quantile(0.25)
iqr_y=dfc.y.quantile(0.75)-dfc.y.quantile(0.25)
iqr_z=dfc.z.quantile(0.75)-dfc.z.quantile(0.25)

dfc=dfc[dfc[target]<(dfc[target].quantile(0.75)+1.5*iqr_target)]
dfc=dfc[dfc[target]>(dfc[target].quantile(0.25)-1.5*iqr_target)]

dfc=dfc[dfc.carat<(dfc.carat.quantile(0.75)+1.5*iqr_carat)]
dfc=dfc[dfc.carat>(dfc.carat.quantile(0.25)-1.5*iqr_carat)]

dfc=dfc[dfc.depth<(dfc.depth.quantile(0.75)+1.5*iqr_depth)]
dfc=dfc[dfc.depth>(dfc.depth.quantile(0.25)-1.5*iqr_depth)]

dfc=dfc[dfc.table<(dfc.table.quantile(0.75)+1.5*iqr_table)]
dfc=dfc[dfc.table>(dfc.table.quantile(0.25)-1.5*iqr_table)]

dfc=dfc[dfc.x<(dfc.x.quantile(0.75)+1.5*iqr_x)]
dfc=dfc[dfc.x>(dfc.x.quantile(0.25)-1.5*iqr_x)]

dfc=dfc[dfc.y<(dfc.y.quantile(0.75)+1.5*iqr_y)]
dfc=dfc[dfc.y>(dfc.y.quantile(0.25)-1.5*iqr_y)]

dfc=dfc[dfc.z<(dfc.z.quantile(0.75)+1.5*iqr_z)]
dfc=dfc[dfc.z>(dfc.z.quantile(0.25)-1.5*iqr_z)]

dfc[target]=np.log1p(dfc[target])

ord=OrdinalEncoder(mapping=[{'col':'cut','mapping':{'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4}},
                            {'col':'color','mapping':{'J':0,'I':1,'H':2,'G':3,'F':4,'E':5,'D':6}},
                            {'col':'clarity','mapping':{'I1':0,'SI2':1,'SI1':2,'VS2':3,'VS1':4,'VVS2':5,'VVS1':6,'IF':7}}])

dfc=ord.fit_transform(dfc)
X_dfc=dfc.drop(target,axis=1)
y_dfc=dfc[target]
rf_rg=RandomForestRegressor(random_state=2,
                            n_estimators=197,
                            max_depth=16,
                            min_samples_split=2,
                            min_samples_leaf=3)

rf_rg.fit(X_dfc,y_dfc)


pickle.dump(rf_rg,open('md.pkl','wb'))