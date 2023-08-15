import os
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

all_data = pd.read_csv("all_data.csv")



def process_data(df):
    df = df.dropna()
    df = df[~df["Order Date"].str.contains("Order Date")]
    
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%y %H:%M")
    df["Month"] = df["Order Date"].dt.month
    
    df["City"] = df["Purchase Address"].str.split(", ").str[1]
    
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
    df['Sales'] = df['Quantity Ordered'] * df['Price Each']

    return df

#calling a function, printing
processed_data = process_data(all_data)
#print (processed_data.head(10))

all_data = processed_data

##### Best month for sales #######

def analyze_sales(all_data):
    monthly_sales = all_data.groupby("Month")["Sales"].sum()
    best_month = monthly_sales.idxmax()

    print("The best month for sales is:", best_month)
    print("Total sales for that month:", monthly_sales[best_month])

    print(all_data.head(10))

# Assuming all_data is your processed DataFrame
#analyze_sales(all_data)


##### displaying info per month #########

def visualize_monthly_sales(all_data):
    monthly_sales = all_data.groupby("Month")["Sales"].sum()
    best_month = monthly_sales.idxmax()

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_sales.index, monthly_sales.values)
    plt.xlabel("Month")
    plt.ylabel("Sales ($)")
    plt.title("Monthly Sales")
    plt.xticks(monthly_sales.index)
    plt.grid(True)
    plt.show()

# Assuming all_data is your processed DataFrame
#visualize_monthly_sales(all_data)


##### displaying info per city #########

def visualize_city_sales(all_data):
    all_data["City"] = all_data["Purchase Address"].apply(lambda x: x.split(", ")[1])

    # Group by city and sum the sales
    city_sales = all_data.groupby("City")["Sales"].sum()

    plt.figure(figsize=(12, 6))
    plt.bar(city_sales.index, city_sales.values)
    plt.xlabel("City")
    plt.ylabel("Sales ($)")
    plt.title("Total Sales by City")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Assuming all_data is your processed DataFrame
#visualize_city_sales(all_data)

###### displaying infor per ad time #######


def visualize_hourly_sales(all_data):
    all_data["Hour"] = all_data["Order Date"].dt.hour

    # Group by hour and sum the sales
    hourly_sales = all_data.groupby("Hour")["Sales"].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_sales.index, hourly_sales.values, marker='o')
    plt.xlabel("Hour")
    plt.ylabel("Sales ($)")
    plt.title("Hourly Sales")
    plt.xticks(hourly_sales.index)
    plt.grid(True)
    plt.show()

# Assuming all_data is your processed DataFrame
#visualize_hourly_sales(all_data)

##### combination of sales #######

def perform_association_analysis(all_data, min_support=0.005, min_lift=1.0, top_n_rules=10):
    order_products = all_data.groupby("Order ID")["Product"].apply(list)
    transactions = order_products.tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values(by=["lift"], ascending=False)

    print(rules.head(top_n_rules))

# Assuming all_data is your processed DataFrame
#perform_association_analysis(all_data, min_support=0.005, min_lift=1.0, top_n_rules=10)

######  what product sold the most   #############

def visualize_product_sales(all_data):
    all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']

    product_sales = all_data.groupby("Product")["Quantity Ordered"].sum()

    plt.figure(figsize=(12, 6))
    product_sales.sort_values(ascending=False).plot(kind="bar")
    plt.xlabel("Product")
    plt.ylabel("Quantity Ordered")
    plt.title("Quantity Ordered for Each Product")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Assuming all_data is your processed DataFrame
visualize_product_sales(all_data)