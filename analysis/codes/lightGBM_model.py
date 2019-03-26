# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

LOAD_MODEL = False

np.random.seed(7)

print('Loading data...')

df = pd.read_csv('dataset_full.csv')

dropcols = ['ACTION', 'ADDRESS', 'BORO', 'BUILDING','CAMIS', 'CRIT_TYPE', 'DBA', 'GRADE',
           'HIST_CRIT_TYPE', 'INSP_DATE','LAST_CRIT_DATE','LAST_INSP_DATE', 'PHONE', 'SCORE', 'STREET','yelp_address1', 'yelp_categories_a', 'yelp_categories_t',
       'yelp_city', 'yelp_day0', 'yelp_day1', 'yelp_day2', 'yelp_day3',
       'yelp_day4', 'yelp_day5', 'yelp_day6', 'yelp_id', 'yelp_latitude', 'yelp_longitude', 'yelp_name', 'yelp_phone',
           'yelp_state',
       'yelp_transactions', 'yelp_url', 'yelp_zip_code','day0_end',
       'day0_start', 'day1_end',
       'day1_start', 'day2_end',
       'day2_start', 'day3_end',
       'day3_start', 'day4_end',
       'day4_start', 'day5_end',
       'day5_start', 'day6_end',
       'day6_start','day0_overnight', 'day1_overnight', 'day2_overnight',
       'day3_overnight', 'day4_overnight', 'day5_overnight', 'day6_overnight','2CRIT_FREQ',
       'NCRIT_FREQ','yelp_is_closed','day0_daily',
       'day1_daily', 'day2_daily', 'day3_daily', 'day4_daily', 'day5_daily',
       'day6_daily','yelp_reviews','gmap_reviews','yelp_reviews_hist','gmap_reviews_hist','ZIPCODE','CRIT','CRIT_FLAG','pumaname','day0_open','CD', 'day1_open',
       'day2_open', 'day3_open', 'day4_open', 'day5_open', 'day6_open','CRIT_TIMES','INSP_TIMES','CUIZ_TYPE','per_in_puma']

# drop unwanted columns
df2 = df.drop(columns = dropcols,axis = 1,inplace = False)
# hash encoding to cuisine type, weight of evidence to zipcode

X = df2.drop(columns = 'CRIT_FLAG2',axis = 1,inplace = False)
y = df2['CRIT_FLAG2']

scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
   
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75)

if LOAD_MODEL:
    # load model from files
    gbm = lgb.Booster(model_file='model.txt')
else:
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'is_unbalance': 'true',         # base rate: 0.30
        'num_leaves': 36,
        "num_threads": 4,
        'learning_rate': 0.02,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 1000,
        'verbose': 0
    }

    print('Starting training...')
    # train model
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_train,  # eval training data
                    categorical_feature=[21])

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')


print('Starting predicting...')

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93]

best_thres, best_score = 0, 0
for th in thresholds:
    y_pred_adjusted = []
    for y in y_pred:
        if y < th:
            y_pred_adjusted.append(0)
        else:
            y_pred_adjusted.append(1)
    cur_score = round(roc_auc_score(y_pred_adjusted, y_test), 4)
    if best_score < cur_score:
        best_score = cur_score
        best_thres = th
    print('The AUC of prediction is:', cur_score, 'with the threshold of', th)
print('***The best AUC is:', best_score, 'with the threshold of', best_thres)


fpr, tpr, thresholds = roc_curve(y, y_pred)

def plotROC(tpr, fpr, label=''):
    """
    Plot ROC curve from tpr and fpr.
    """
    plt.plot(fpr, tpr, label=label)
    plt.legend()
    plt.ylabel('True positive rate.')
    plt.xlabel('False positive rate')
    plt.show()

plotROC(tpr, fpr, label='Test')
