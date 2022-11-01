import pandas as pd
import numpy as np
# 导入lightGBM训练数据的类库
import lightgbm as lgb
# 导入交叉验证使用的类库
from sklearn.model_selection import KFold
# 导入转换编码的类库
from sklearn.preprocessing import LabelEncoder
# 导入误差函数rmse的计算类库
from sklearn.metrics import mean_squared_error
import math

df_train = pd.read_csv('train_data.csv', encoding='gbk')
df_test = pd.read_csv('A_test_data.csv', encoding='gbk')
data = pd.concat([df_train, df_test], axis=0)

for col in data.columns:
    if col not in ['ret', 'time', 'id']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d')
data['month']=data['time'].dt.month
data['day']=data['time'].dt.day
data['weekday']=data['time'].dt.weekday
train=data[data['ret'].notnull()]
test=data[data['ret'].isnull()]
feature=[x for x in train.columns if x not in ['ret', 'time', 'id']]


#lgb
clf = lgb.LGBMRegressor(
    learning_rate=0.05,
    n_estimators=50230,
    num_leaves=31,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    metric='rmse'
)


train_x=train[feature]

target=train['ret']
test_x=test[feature]

oof1 = np.zeros(len(train))
answers = []
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True,random_state=2000)


for fold_n, (train_index, valid_index) in enumerate(folds.split(train_x)):
    X_train, X_valid = train_x.iloc[train_index], train_x.iloc[valid_index]
    y_train, y_valid = target[train_index], target[valid_index]
    clf.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],verbose=100,early_stopping_rounds=200)
    y_pre=clf.predict(X_valid)
    oof1[valid_index]=y_pre
    y_pred_valid = clf.predict(test_x)
    answers.append(y_pred_valid)


lgb_pre=sum(answers)/n_fold


print('score-----------',  (1/((math.sin(math.atan(np.sqrt(mean_squared_error(oof1,target)))))+1)))

sub=df_test[['id']]
sub['ret']=lgb_pre
sub.to_csv('submit.csv',index=False)
