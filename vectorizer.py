import pandas as pd
from pandas  import get_dummies
import numpy as np

df = pd.read_csv("BlackFriday.csv")
df.drop(['Product_Category_1', 'Product_Category_2', 'Product_Category_3'], axis=1, inplace=True)

purchase_df = df[['User_ID', 'Product_ID']].groupby('User_ID')['Product_ID'].apply(list)

df.drop('Product_ID', axis=1, inplace=True)
max_purchase = max(df.Purchase)
df['Purchase'] = df['Purchase'].apply(lambda k: k / max_purchase)

dummies = get_dummies(df)
dummies = dummies.groupby('User_ID').first()

vectors = {}
for i, uid in enumerate(dummies.index):
    vectors[uid] = np.array(dummies.iloc[i])

print('User: 1000001 -> ', vectors[1000001])
print("Purchases:")
for item in purchase_df.loc[1000001]:
    print(item)
