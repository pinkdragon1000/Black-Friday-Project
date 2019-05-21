import pandas as pd
from pandas import get_dummies
import numpy as np
import logging

FORMAT = "[%(asctime)s] - [%(levelname)s] - [%(funcName)s] - %(message)s"
logging.basicConfig(level=20, format=FORMAT)


def get_data():
    df = pd.read_csv("BlackFriday.csv")
    logging.info("Raw data loaded")

    df.drop(
        ["Product_Category_1", "Product_Category_2", "Product_Category_3"],
        axis=1,
        inplace=True,
    )

    purchase_df = (
        df[["User_ID", "Product_ID"]].groupby("User_ID")["Product_ID"].apply(list)
    )

    df.drop("Product_ID", axis=1, inplace=True)

    df["Occupation"] = df["Occupation"].astype(str)

    gbc = list(df.columns[:-1])
    grouped = df.groupby(gbc).sum().reset_index()
    grouped["Purchase"] = grouped.Purchase / max(grouped.Purchase)

    dummies = get_dummies(grouped)

    dummies.index = dummies.User_ID
    dummies.drop("User_ID", inplace=True, axis=1)

    logging.info("Data cleaned")

    return purchase_df, dummies


def get_vectors(df):
    logging.info("Collecting vectors")
    vectors = {}
    for i, uid in enumerate(df.index):
        vectors[uid] = np.array(df.iloc[i])
    logging.info("Vectors ready")
    return vectors


def main():
    purchase_df, user_df = get_data()
    vectors = get_vectors(user_df)

    print("User: 1000001 -> \n", vectors[1000001])
    print("Purchases:")
    for item in purchase_df.loc[1000001]:
        print(item)


if __name__ == "__main__":
    main()
