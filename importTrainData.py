#%%
import numpy as np
import pandas as pd
import pickle
import category_encoders as ce

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

db_genba = pd.read_csv("train_genba.tsv",encoding="utf8",sep="\t")
db_goto = pd.read_csv("train_goto.tsv",encoding="utf8",sep="\t")

#%%
db = pd.merge(db_genba, db_goto, on='pj_no', how='left')

#%%
Y = db["keiyaku_pr"]
X = db.drop("keiyaku_pr",axis=1)

X = X.fillna(0)
Y = Y.fillna(0)

#%%

#タイプがobject(文字列)の列list
list_cols = X.columns[X.dtypes == "object"]
list_cols = list(list_cols)

#%%

# OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。
ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')
X = ce_oe.fit_transform(X)



#%%

N_train = int(len(X.index) * 0.8)
N_test = len(X.index) - N_train

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=N_test,shuffle = False)

tuned_parameters = [{'n_estimators':[200,300]}]
clf = GridSearchCV(
    RandomForestClassifier(),
    tuned_parameters,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

clf.fit(X_train,list(Y_train))

clf = clf.best_estimator_


#%%
