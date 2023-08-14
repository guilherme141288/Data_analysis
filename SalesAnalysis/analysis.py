import os
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#path = r"C:\Users\Guilherme Bertuol\Documents\data_analysis\exe_02\SalesAnalysis\Sales_Data" # Update with the path to your directory containing CSV files
#files = [file for file in os.listdir(path) if file.endswith('.csv')]  # Select only CSV files

#all_months_data = pd.concat([pd.read_csv(os.path.join(path, file)) for file in files], ignore_index=True)

#all_months_data.to_csv("all_data_combined.csv", index=False)


all_data = pd.read_csv("all_data.csv")


#######  Cleaning data ##########
df = pd.DataFrame(all_data)
all_data = all_data.dropna()

# Filter out rows with header names or invalid date values
all_data = all_data[~all_data["Order Date"].str.contains("Order Date")]

# Convert "Order Date" column to datetime with specified format
all_data["Order Date"] = pd.to_datetime(all_data["Order Date"], format="%m/%d/%y %H:%M")

# Extract month from "Order Date" and create a new column
all_data["Month"] = all_data["Order Date"].dt.month

# Extract city from "Purchase Address" using .str.split() and .str[1]
all_data["City"] = all_data["Purchase Address"].str.split(", ").str[1]

all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'] , errors='coerce')
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'] , errors='coerce')



############ best month for sales ############


all_data['Sales'] = all_data['Quantity Ordered'].astype('float') * all_data['Price Each'].astype('float')
#monthly_sales = all_data.groupby("Month")["Sales"].sum()
#best_month = monthly_sales.idxmax()


#print("The best month for sales is:", best_month)
#print("Total sales for that month:", monthly_sales[best_month])

#print(all_data.head(50))

##### displaying info per month #########

#plt.figure(figsize=(10, 6))
#plt.bar(monthly_sales.index, monthly_sales.values)
#plt.xlabel("Month")
#plt.ylabel("Sales ($)")
#plt.title("Monthly Sales")
#plt.xticks(monthly_sales.index)
#plt.grid(True)
#plt.show()

##### displaying info per city #########

#all_data["City"] = all_data["Purchase Address"].apply(lambda x: x.split(", ")[1])

# Group by city and sum the sales
#city_sales = all_data.groupby("City")["Sales"].sum()

#plt.figure(figsize=(12, 6))
#plt.bar(city_sales.index, city_sales.values)
#plt.xlabel("City")
#plt.ylabel("Sales ($)")
#plt.title("Total Sales by City")
#plt.xticks(rotation=45, ha="right")
#plt.tight_layout()
#plt.show()


###### displaying infor per ad time #######


#all_data["Hour"] = all_data["Order Date"].dt.hour

# Group by hour and sum the sales
#hourly_sales = all_data.groupby("Hour")["Sales"].sum()

# Plotting using Matplotlib
#plt.figure(figsize=(10, 6))
#plt.plot(hourly_sales.index, hourly_sales.values, marker='o')
#plt.xlabel("Hour")
#plt.ylabel("Sales ($)")
#plt.title("Hourly Sales")
#plt.xticks(hourly_sales.index)
#plt.grid(True)
#plt.show()

##### combination of sales #######

# Extract order ID and product names
#order_products = all_data.groupby("Order ID")["Product"].apply(list)

# Convert product lists into transaction data
#transactions = order_products.tolist()

# Perform association analysis

#te = TransactionEncoder()
#te_ary = te.fit(transactions).transform(transactions)
#df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets
#frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)

# Generate association rules
#rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
#rules = rules.sort_values(by=["lift"], ascending=False)

# Display the top 10 association rules
#print(rules.head(10))


######  what product sold the most   #############


# Calculate the 'Sales' column after ensuring numeric data types
all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']

# Group by product and sum the quantities ordered
product_sales = all_data.groupby("Product")["Quantity Ordered"].sum()

# Plotting using Matplotlib
plt.figure(figsize=(12, 6))
product_sales.sort_values(ascending=False).plot(kind="bar")
plt.xlabel("Product")
plt.ylabel("Quantity Ordered")
plt.title("Quantity Ordered for Each Product")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
 