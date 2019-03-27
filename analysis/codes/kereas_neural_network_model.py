#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU, PReLU

np.random.seed(7)

df = pd.read_csv('dataset_full.csv')

dropcols = ['ACTION', 'ADDRESS', 'BORO', 'BUILDING','CAMIS', 'CRIT_TYPE', 'DBA', 'GRADE', 'HIST_CRIT_TYPE',
            'INSP_DATE','LAST_CRIT_DATE','LAST_INSP_DATE', 'PHONE', 'SCORE', 'STREET','yelp_address1',
            'yelp_categories_a', 'yelp_categories_t', 'yelp_city', 'yelp_day0', 'yelp_day1', 'yelp_day2', 'yelp_day3',
            'yelp_day4', 'yelp_day5', 'yelp_day6', 'yelp_id', 'yelp_latitude', 'yelp_longitude', 'yelp_name',
            'yelp_phone', 'yelp_state', 'yelp_transactions', 'yelp_url', 'yelp_zip_code','day0_end', 'day0_start',
            'day1_end', 'day1_start', 'day2_end', 'day2_start', 'day3_end', 'day3_start', 'day4_end', 'day4_start',
            'day5_end', 'day5_start', 'day6_end', 'day6_start', 'day0_overnight', 'day1_overnight', 'day2_overnight',
            'day3_overnight', 'day4_overnight', 'day5_overnight', 'day6_overnight','2CRIT_FREQ', 'NCRIT_FREQ',
            'yelp_is_closed','day0_daily', 'day1_daily', 'day2_daily', 'day3_daily', 'day4_daily', 'day5_daily',
            'day6_daily','yelp_reviews','gmap_reviews','yelp_reviews_hist','gmap_reviews_hist','ZIPCODE','CRIT',
            'CRIT_FLAG','pumaname','day0_open','CD', 'day1_open', 'day2_open', 'day3_open', 'day4_open', 'day5_open',
            'day6_open','CRIT_TIMES','INSP_TIMES','CUIZ_TYPE','per_in_puma']

df2 = df.drop(columns = dropcols,axis = 1,inplace = False)
# hash encoding to cuisine type, weight of evidence to zipcode

# no text data:
# notext = df2.columns.values[-7671:-47]
# df2.drop(columns = notext, axis = 1,inplace = True)

X = df2.drop(columns = 'CRIT_FLAG2',axis = 1,inplace = False)
Y = df2['CRIT_FLAG2']
   
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
   
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)

model = Sequential()
model.add(Dense(50, input_dim=7708, activation='relu',kernel_regularizer = regularizers.l2(0.001)))
model.add(LeakyReLU(alpha = .001))
model.add(Dense(40, activation='relu',kernel_regularizer = regularizers.l2(0.001)))
model.add(LeakyReLU(alpha = .001))
model.add(Dense(1, activation='softmax'))
   
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y_train, epochs=150, batch_size=20)
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))








