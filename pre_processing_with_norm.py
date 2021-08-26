import numpy as np
from pickle import dump
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def process_with_norm (data,ch):

    # fill null with values
    data['Directors'] = data['Directors'].fillna('NONE')
    data['Genres'] = data['Genres'].fillna('NONE')
    data['Country'] = data['Country'].fillna('NONE')
    data['Language'] = data['Language'].fillna('NONE')
    data['Runtime'] = data['Runtime'].fillna(100)
    if ch==1:
        data['rate'] = data['rate'].fillna('Intermediate')
    else:
        data['IMDb'] = data['IMDb'].fillna(6.6)

    #label encoding
    X = data['Directors']
    t_x=[]
    for i in X:
        t_x.append(i)

    x_label_encoder = LabelEncoder()
    t_x = x_label_encoder.fit_transform(t_x)
    data['Directors'] = t_x
    data['Directors'] = normarilzation(data['Directors'])

    X = data['Language']
    t_x = []
    for i in X:
        t_x.append(i)

    x_label_encoder = LabelEncoder()
    t_x = x_label_encoder.fit_transform(t_x)
    data['Language'] = t_x
    data['Language'] = normarilzation(data['Language'])

    X = data['Country']
    t_x = []
    for i in X:
        t_x.append(i)

    x_label_encoder = LabelEncoder()
    t_x = x_label_encoder.fit_transform(t_x)
    data['Country'] = t_x
    data['Country'] = normarilzation(data['Country'])

    # manual 1 hot encoding

    X=data['Genres']

    df=pd.DataFrame(data)
    df.insert(10,"Action",0)
    df.insert(11, "Adventure", 0)
    df.insert(12, "Sci-Fi", 0)
    df.insert(13, "Thriller", 0)
    df.insert(14, "Comedy", 0)
    df.insert(15, "Western", 0)
    df.insert(16, "Animation", 0)
    df.insert(17, "Family", 0)
    df.insert(18, "Biography", 0)
    df.insert(19, "Drama", 0)
    df.insert(20, "Music", 0)
    df.insert(21, "War", 0)
    df.insert(22, "Crime", 0)
    df.insert(23, "Fantasy", 0)
    df.insert(24, "Romance", 0)
    df.insert(25, "Sport", 0)
    df.insert(26, "Mystery", 0)
    df.insert(27, "History", 0)
    df.insert(28, "Documentary", 0)
    df.insert(29, "Musical", 0)
    df.insert(30, "News", 0)
    df.insert(31, "Horror", 0)
    df.insert(32, "Reality - TV", 0)
    df.insert(33, "Short", 0)
    data = df

    X = data['Genres']
    X_AC =  data['Action']
    X_Ad = data['Adventure']
    X_SF = data['Sci-Fi']
    X_TH = data['Thriller']
    X_CO = data['Comedy']
    X_WE = data['Western']
    X_AN = data['Animation']
    X_FA = data['Family']
    X_BG = data['Biography']
    X_DR = data['Drama']
    X_MU = data['Music']
    X_W = data['War']
    X_CR = data['Crime']
    X_FAN = data['Fantasy']
    X_RO = data['Romance']
    X_SP = data['Sport']
    X_MY = data['Mystery']
    X_HI = data['History']
    X_DO = data['Documentary']
    X_MUL = data['Musical']
    X_N = data['News']
    X_HO = data['Horror']
    X_RT = data['Reality - TV']
    X_SH = data['Short']

    for i in range(len(X)):
        X_elem = X[i].split(',')
        for j in X_elem:
            if j == "Action":
                X_AC[i]=1
            if j == "Adventure":
                X_Ad[i]=1
            if j == "Sci-Fi":
                X_SF[i] = 1
            if j == "Thriller":
                X_TH[i]=1
            if j == "Western":
                X_WE[i]=1
            if j == "Animation":
                X_AN[i]=1
            if j == "Family":
                X_FA[i]=1
            if j == "Biography":
                X_BG[i]=1
            if j == "Drama":
                X_DR[i] = 1
            if j == "Music":
                X_MU[i]=1
            if j == "War":
                X_W[i]=1
            if j == "Crime":
                X_CR[i]=1
            if j == "Fantasy":
                X_FAN[i]=1
            if j == "Romance":
                X_RO[i]=1
            if j == "Sport":
                X_SP[i]=1
            if j == "Mystery":
                X_MY[i]=1
            if j == "History":
                X_HI[i]=1
            if j == "Documentary":
                X_DO[i]=1
            if j == "Musical":
                X_MUL[i]=1
            if j == "News":
                X_N[i]=1
            if j == "Horror":
                X_HO[i]=1
            if j == "News":
                X_N[i]=1
            if j == "Reality - TV":
                X_RT[i]=1
            if j == "Short":
                X_SH[i]=1

    data['Action'] =X_AC
    data['Adventure'] =X_Ad
    data['Sci-Fi'] =X_SF
    data['Thriller'] =X_TH
    data['Comedy'] =X_CO
    data['Western'] =X_WE
    data['Animation'] =  X_AN
    data['Family'] =X_FA
    data['Biography'] =X_BG
    data['Drama'] =X_DR
    data['Music'] =X_MU
    data['War'] =X_W
    data['Crime'] =X_CR
    data['Fantasy'] =X_FAN
    data['Romance'] =X_RO
    data['Sport'] =X_SP
    data['Mystery'] =X_MY
    data['History'] =X_HI
    data['Documentary'] =X_DO
    data['Musical'] =X_MUL
    data['News'] =X_N
    data['Horror'] =X_HO
    data['Reality - TV'] =X_RT
    data['Short'] =X_SH

    #print(data.to_string())

    #scaling
    data['Year']=normarilzation(data['Year'])

    X= data['Age']
    res=[]
    for i in X:
        if i is np.nan or i == "all":
            i="13+"
        X_elem = i.split('+')
        del X_elem[1]
        X_elem[0]=int(X_elem[0])
        res.append(X_elem[0])

    for i in range(len(res)):
        X[i] = (res[i] - min(res)) / (max(res) - min(res))

    data['Age'] = X


    X = data['Rotten Tomatoes']
    res = []
    for i in X:
        if i is np.nan :
            i = "60%"
        X_elem = i.split('%')
        del X_elem[1]
        X_elem[0] = int(X_elem[0])
        res.append(X_elem[0])

    for i in range(len(res)):
        X[i] = (res[i] - min(res)) / (max(res) - min(res))

    X = (X - min(X)) / (max(X) - min(X))
    data['Rotten Tomatoes'] = X

    data['Runtime'] = normarilzation(data['Runtime'])

    #drop colums
    del data['Type']
    del data['Title']
    del data['Genres']
    if ch == 1:
        dump(data, open('c_data_N.pkl', 'wb'))
    else:
        dump(data, open('data_N.pkl', 'wb'))


    return data


def normarilzation(X):
    X = (X - min(X)) / (max(X) - min(X))
    return X
