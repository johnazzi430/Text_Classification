


from __future__ import division


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os
import csv
import math
import re
import sklearn
import theano
import keras
from scipy import sparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
sns.set(color_codes=True)


### download review data from here https://www.kaggle.com/zynicide/wine-reviews/data


df_wine = pd.read_csv('winemag-data-130k-v2.csv')
data = df_wine.drop_duplicates('description')
data = data[pd.notnull(data.price)]
df_wine=data
#### Simple Bag of Words

tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
tfidf_transformer = sklearn_tfidf.fit(df_wine['description'].values)
tfidf_out = tfidf_transformer.transform(df_wine['description'].values)
vocab = np.array(tfidf_transformer.get_feature_names())

def context():
    return

#### Outdated does the same thing as tfidfvectorizer but not as refined
def BoW(text):
    context.word_index = []
    for descript in text:
        descript = descript.split()
        for wrd in descript:
            if wrd.lower() not in context.word_index:
                context.word_index.append(wrd.lower())
            else:
                pass

    context.word_index.sort()
    context.word_index = np.array(context.word_index)
    mat_out = []
    for desc in text:
        vect_out = []
        descript = [wrd.lower() for wrd in desc.split()]
        for wrd in context.word_index:
            vect_out.append(descript.count(wrd))
        mat_out.append(vect_out)
    mat_out = np.asarray(mat_out)

    data_in = mat_out
    context.common = np.array((data_in.sum(0), context.word_index))
    context.common_s = context.common[:, data_in.sum(0).argsort()]
    context.keep = []
    for i in range(0, len(context.word_index)):
        if int(context.common[0, :][i]) / len(context.word_index) < 0.15:
            context.keep.append(i)
    context.word_index_new = context.word_index[context.keep]
    data_out = data_in[:, context.keep]
    return data_out

### not used
def prepare_x(matrix_out):
    x_data =[]
    for row in matrix_out:
        maxx=np.max(row)
        x_data.append(row / maxx)
    x_data = np.array(x_data)
    return x_data

def prepare_y_cat(data_in):
    index=[]
    data_out =[]
    data_in_s = data_in
    data_in_s.sort()
    for n in data_in_s:
        if n not in index:
            index.append(n)
    for j in data_in:
        whr = np.where ( np.asarray(index) == j)
        data_out.append(whr[0][0])
    context.dict_name = index
    return data_out

def prepare_y(data_in):
    z = data_in.tolist()
    z_max = np.max(z)
    z_min = np.min(z)
    z = (z - z_min) / (z_max - z_min)
    z_exp = [math.exp(i) for i in z]
    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    softmax = z
    return softmax

################


#### Define Goal of code:

#       Predict, based just on description if a wine is greater than or less than a rating of 95 points

### points distribution in dataset
sns.distplot(df_wine['points'].values)
plt.show()


#### bespoke mean needed to avoid crashing on large datasets
def largemean(data):
    calcout = []
    for x in range(500,data.shape[0],500):
        calcout.append(np.mean(data[x-500:x,:],0))
    out = np.mean(np.array(calcout),0)
    return out


#### sort
tfidf_in = tfidf_out[np.argsort(df_wine['points'].values)]
points = np.sort(df_wine['points'].values)

data = tfidf_in[np.where(points > 95)].toarray()
data_op = tfidf_in[np.where(points <= 95)].toarray()
mean_up = largemean(data)
mean_down = largemean(data_op)
data_mean_relative = np.mean(data,0) - np.mean(data_op,0)
data_argsort = np.argsort(data_mean_relative)[::-1]


top40 = vocab[data_argsort][:40]

print('top 40 words that describe a wine which will exceed 95 points')
for x in range(0,len(top40) , 4):
    print(top40[x],',', top40[x+1],',',top40[x+2],',',top40[x+3])

### Create Y set
Y=[]
for pnt in points:
    if pnt <= 95:
        Y.append([0,1])
    else:
        Y.append([1,0])

Y = np.asarray(Y)
X = tfidf_in.toarray()

##### describe


#X_ = prepare_x(matrix_out)
#X_ = sparse.csr_matrix(X_)

#Y_ = df_wine['points'].values
#Y_ = np.asarray(prepare_y(df_wine['points']))
#Y_ = prepare_y_cat(df_wine['variety'].values)
#Y_e = Y_.reshape(-1,1)
#enc = sklearn.preprocessing.OneHotEncoder(sparse = False)
#Y_e = enc.fit_transform(Y_e)

#X_ = sklearn_representation

Y_=Y_e
Y_ = Y_.reshape(-1,1)

pct = int(0.85 * Y_.shape[0])
X_train = X_[:pct]
X_test = X_[pct:]
Y_train = Y_[:pct]
Y_test = Y_[pct:]

#### NOTE: what if Y data is the type?


##### apply some machine learning


model = Sequential()
model.add(Dense(1000 , input_dim=X_train.shape[1], activation='softmax'))
model.add(Dropout(0.25))
model.add(Dense(1060, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(Y_train.shape[1], activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'] )
#model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['accuracy'] )
model.fit(X_train, Y_train, epochs=4, batch_size=100)

score = model.evaluate(X_test, Y_test, verbose=True)

trial_X = X_test[8].reshape(1,X_test[5].shape[0])
trial = model.predict(trial_X)

print( 'model accuracy:' , round(score[1],3) , '%')


#df_X = pd.DataFrame(X_)
#corr_matrix=df_X.corr()

#p1 = sns.heatmap(pd.DataFrame(X_))

####### Define words commonly used
