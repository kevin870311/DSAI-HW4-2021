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



sns.countplot(data = orders_df, x ='order_dow')
plt.xlabel('Hour of the day')
plt.ylabel('Orders Count ')
plt.title('Orders by Hour of the Day')
plt.show()

sns.countplot(data = orders_df, x ='order_hour_of_day',palette='viridis')
plt.xlabel('Hour of the day ')
plt.ylabel('Orders Count ')
plt.title('Orders by Hour of the day')
plt.show()

plt.subplots(figsize=(25, 18))
sns.countplot(data =orders_df, x='days_since_prior_order',palette='viridis')
plt.xlabel('Number of days since prior order')
plt.ylabel('Number of orders')
plt.title('Distribution of Reoders')
plt.show()


plt.figure(figsize=(12, 8))
tmp=pd.DataFrame(orders_df['user_id'].value_counts().values,columns=['user_correspoding_samples'])
tmp=tmp['user_correspoding_samples'].value_counts()
chart = sns.barplot(x=tmp.index,y=tmp.values)
plt.tick_params(axis='x', labelsize=7)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

plt.figure(figsize=(12, 4))
sns.countplot(x="order_dow", data=orders_df)
plt.xlabel("Day of week")
plt.xticks([i for i in range(7)], ["Sat.", "Sun.", "Mons", "Tues.", "wed.", "Thur.", "Fri."])
plt.title("The frequency in differen day of week")
plt.show()

df = pd.merge(order_products_prior_df, prducts_df, on ='product_id', how='left')
df = pd.merge(df, aisles_df, on ='aisle_id', how='left')
df = pd.merge(df, departments, on ='department_id', how='left')
df = pd.merge(df, orders_df, on ='order_id', how='left' )

bs= df.groupby('department')['order_id'].count().reset_index()
bs.columns = ['department','total_orders']

bs_plot = bs.sort_values(by='total_orders', ascending=False)

plt.subplots(figsize=(20,10))
plt.xticks(rotation = 'vertical')
sns.barplot(bs_plot.department, bs_plot.total_orders)
plt.xlabel('Departments')
plt.ylabel('Total Orders')
plt.title('Best selling Departments', fontsize= 20)
plt.show()