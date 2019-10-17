from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import shelve

def get_time_series(file, length):
    df = pd.read_csv(file)
    t_prev = 0.0
    X = []
    x = []
    y = []

    for row in df.values:
        t, a, v = row[0], row[1: 14], row[14: ]
        if x:
            if round(t - t_prev) < 35:
                x.append(a)
                if len(x) == length:
                    X.append(x)
                    y.append(v)
                    x = x[1: ]
            else:
                x = [a]
        else:
            x.append(a)
            if len(x) == length:
                X.append(x)
                y.append(v)
                x = x[1: ]

        t_prev = t

    return X, y

def prepr(length, test_size=0.1, shuff=True):
    X = []
    y = []

    for i in range(sum(f_name.startswith('merge') for f_name in os.listdir('extracted_data'))):
        X_, y_ = get_time_series('extracted_data/merge_{}.csv'.format(i), length)
        db = shelve.open('extracted_data/default_points_{}.txt'.format(i), 'r')
        default_points = np.array(db['default_face']).ravel()
        db.close()
        y_ = y_ - default_points
        y_ = np.array(y_)
        X.extend(X_)
        y.extend(y_)
    print('asd', y[0])
    for i, x in enumerate(X):
        tmp = []
        for mfcc in x:
            tmp.extend(mfcc)
        X[i] = tmp
    X = np.array(X)

    #with shelve.open('default_points.txt', 'r') as db:
    #    default_points = np.array(db['default_face']).ravel()

    for i, y_ in enumerate(y):
    #    y[i] = np.array(y_) - default_points
        y[i] = y[i][96:]
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuff, random_state=42)

    # нормализация звуков
    scaler = StandardScaler()
    #print(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    prepr(length=5)