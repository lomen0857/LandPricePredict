#%%
import numpy as np
import pandas as pd

db_genba = pd.read_csv("train_genba.tsv",encoding="utf8",sep="\t")
db_goto = pd.read_csv("train_goto.tsv",encoding="utf8",sep="\t")

#%%
db = pd.merge(db_genba, db_goto, on='pj_no', how='left')

#%%
Y = db["keiyaku_pr"]
X = db.drop("keiyaku_pr",axis=1)

#%%
