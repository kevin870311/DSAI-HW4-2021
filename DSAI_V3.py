import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

path='D:/研究所/'
aisles_df = pd.read_csv(path+'aisles.csv')
departments = pd.read_csv(path + 'departments.csv')
prducts_df  = pd.read_csv(path + 'products.csv')
orders_df  = pd.read_csv(path + 'orders.csv')
order_products_train_df  = pd.read_csv(path + 'order_products__train.csv')
order_products_prior_df  = pd.read_csv(path + 'order_products__prior.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')

aa = orders_df.head(50)
a1 = set(order_products_train_df)
a2 = set(order_products_prior_df)
a1.issubset(a2)
aa = order_products_prior_df.head(50)
# Order view
user_fea = pd.DataFrame()
user_fea['user_id'] = orders_df['user_id'].unique()
user_fea = user_fea[['user_id']].sort_values('user_id')
user_fea['user_orderid_count']=orders_df.groupby('user_id')['order_id'].count().values #每個user總購買次數

user_fea['user_days_since_prior_order_mean'] = orders_df.groupby('user_id')['days_since_prior_order'].mean().values
user_fea['user_days_since_prior_order_max'] = orders_df.groupby('user_id')['days_since_prior_order'].max().values


user_fea['user_order_dow_mode'] = orders_df.groupby('user_id')['order_dow'].apply(lambda x: x.mode()[0]).values
user_fea['user_order_hour_of_day_mode'] = orders_df.groupby('user_id')['order_hour_of_day'].apply(lambda x: x.mode()[0]).values

aa = orders_df.head(50)
aaa = order_products_prior_df.head(100)

order_products_prior_df = order_products_prior_df.merge(orders_df,on='order_id',how='left')

user_fea['user_product_nunique'] = order_products_prior_df.groupby('user_id')['product_id'].nunique().sort_index().values
user_fea['user_product_count']=order_products_prior_df.groupby('user_id')['product_id'].count().sort_index().values
user_fea['user_product_orderid_ratio'] = user_fea['user_product_count'] / user_fea['user_orderid_count']

tmp=order_products_prior_df.groupby(['user_id','order_id'])['add_to_cart_order'].max().reset_index()
user_fea['user_add_to_cart_order_max'] = tmp.groupby('user_id')['add_to_cart_order'].max().sort_index().values
user_fea['user_add_to_cart_order_mean'] = tmp.groupby('user_id')['add_to_cart_order'].mean().sort_index().values

user_fea['user_reordered_sum'] = order_products_prior_df.groupby('user_id')['reordered'].sum().sort_index().values
user_fea['user_reordered_mean'] = order_products_prior_df.groupby('user_id')['reordered'].mean().sort_index().values


#Product
product_fea=pd.DataFrame()
product_fea['product_id']=order_products_prior_df['product_id'].unique()
product_fea= product_fea.sort_values('product_id')

product_fea['product_count'] = order_products_prior_df.groupby('product_id')['user_id'].count().sort_index().values
product_fea['product_order_nunqiue'] = order_products_prior_df.groupby('product_id')['order_id'].nunique().sort_index().values
product_fea['product_add_to_cart_order_mean'] = order_products_prior_df.groupby('product_id')['add_to_cart_order'].mean().sort_index().values

product_fea['product_dow_mode'] = order_products_prior_df.groupby('product_id')['order_dow'].apply(lambda x: x.mode()[0]).sort_index().values
product_fea['product_hour_of_day_mode'] = order_products_prior_df.groupby('product_id')['order_hour_of_day'].apply(lambda x: x.mode()[0]).sort_index().values

product_fea['product_reordered_mean'] = order_products_prior_df.groupby('product_id')['reordered'].mean().sort_index().values
product_fea['product_reordered_sum'] = order_products_prior_df.groupby('product_id')['reordered'].sum().sort_index().values 

#User + Product
order_products_prior_df['user_product'] = order_products_prior_df['user_id'].values * 10**5 + order_products_prior_df['product_id'].values

userXproduct_fea=pd.DataFrame()

userXproduct_fea['user_product'] = order_products_prior_df['user_product'].unique() 
userXproduct_fea = userXproduct_fea[['user_product']].sort_values('user_product')

userXproduct_fea['user_product_reordered_sum'] = order_products_prior_df.groupby('user_product')['reordered'].sum().sort_index().values 

userXproduct_fea['user_product_add_to_cart_order_sum'] = order_products_prior_df.groupby('user_product')['add_to_cart_order'].sum().sort_index().values 
userXproduct_fea['user_product_add_to_cart_order_mean'] = order_products_prior_df.groupby('user_product')['add_to_cart_order'].mean().sort_index().values 

userXproduct_fea['user_product_order_nunique'] = order_products_prior_df.groupby('user_product')['order_id'].nunique().sort_index().values 

userXproduct_fea.head()

#Construct training set and validation set 

orders_prior_data = orders_df.loc[orders_df.eval_set == 'prior']
orders_train_data = orders_df.loc[orders_df.eval_set == 'train'] 
orders_test_data  = orders_df.loc[orders_df.eval_set == 'test' ] 

priors = order_products_prior_df.merge(orders_prior_data, on =['order_id'], how='left')
trains = order_products_train_df.merge(orders_train_data, on =['order_id'], how='left')

user_product = order_products_prior_df[['user_id','product_id']].copy()
user_product['user_X_product'] = user_product['user_id'].values* 10**5  + user_product['product_id'].values
train_user_X_product = trains['user_id'].values* 10**5 + trains['product_id'].values

user_product = user_product.drop_duplicates(subset=['user_X_product'], keep = 'last')  #找出user_X_product一樣的表示有重複購買

test_user  = orders_test_data['user_id']
train_user = orders_train_data['user_id']

user_product['label'] = 0
train_data = user_product.loc[user_product.user_id.isin(train_user)]
train_data.loc[train_data.user_X_product.isin(train_user_X_product), 'label'] = 1 

train_data = train_data.merge(orders_train_data,on ='user_id', how='left')
test_data  = user_product.loc[user_product.user_id.isin(test_user)]
test_data = test_data.merge(orders_test_data,on ='user_id', how='left')

train_data = train_data.merge(user_fea, on='user_id', how='left')
train_data = train_data.merge(product_fea, on='product_id', how='left')
train_data = train_data.merge(userXproduct_fea, left_on='user_X_product', right_on='user_product', how='left')
train_data = train_data.merge(prducts_df, on='product_id', how= 'left')

test_data = test_data.merge(user_fea, on='user_id', how='left')
test_data = test_data.merge(product_fea, on='product_id', how='left')
test_data = test_data.merge(userXproduct_fea, left_on='user_X_product', right_on='user_product', how='left')
test_data  =  test_data.merge(prducts_df, on='product_id', how= 'left')

aa = train_data.head(100)

fea_not_need = ['user_id','user_X_product','order_id','eval_set','product_name','user_product_last_order_num','label']
feature_cols = [col for col in train_data.columns if col not in fea_not_need]
label_cols = 'label'

def validation_sample(order_ids, frac = 0.2):
    import random
    sample_number = int(frac * len(order_ids))
    sample_val_order = random.sample( order_ids , sample_number) 
    sample_train_order = list(set(order_ids) - set(sample_val_order))
    return sample_train_order,sample_val_order
sample_train_order,sample_val_order = validation_sample(list(train_data['order_id'].unique()))
train = train_data.loc[train_data.order_id.isin(sample_train_order)]
val   = train_data.loc[train_data.order_id.isin(sample_val_order)]

#Model
import lightgbm as lgb
from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label() 
    y_hat = np.round(y_hat >= 0.2) # scikits f1 doesn't like probabilities 
    return 'f1', f1_score(y_true, y_hat), True

d_train = lgb.Dataset(train[feature_cols], label=train[label_cols].values)   
d_val   = lgb.Dataset(val[feature_cols], label=val[label_cols].values)    
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 2 ** 5,
    'max_depth': 10,
    'learning_rate':0.1,
   # 'feature_fraction': 0.9,
   # 'bagging_fraction': 0.9,    
    'bagging_freq': 5
}
ROUNDS = 120 
bst = lgb.train(params, d_train, ROUNDS, valid_sets=[d_train,d_val], feval =lgb_f1_score,verbose_eval=10)
del d_train

pred = bst.predict(test_data[feature_cols])
test_data['pred'] = pred 
test_data['product_id'] = test_data['product_id'].astype(str)

order_product = {}
for order_id, val, product_id in test_data[['order_id','pred','product_id']].values:
    if order_id in order_product.keys():
        if val >= 0.21:
            if order_product[order_id] == '':
                order_product[order_id] = str(product_id )
            else:
                order_product[order_id] += ' ' + str(product_id )
    else:
        order_product[order_id] = ''
        if val >= 0.21:
            order_product[order_id] = str(product_id ) 

sub = pd.DataFrame.from_dict(order_product, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.loc[sub.products =='', 'products'] = 'None'
sub.to_csv('simple_fe_0.21.csv',index = None)
