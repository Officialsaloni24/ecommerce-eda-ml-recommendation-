#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ================================
# E-COMMERCE EDA PROJECT (ERROR-FREE)
# ================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")

# 2. Load Dataset
df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

# 3. Initial Inspection
print("Dataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe(include="all"))

# 4. Data Cleaning

# Drop missing CustomerID
if 'CustomerID' in df.columns:
    df = df.dropna(subset=['CustomerID'])

# Remove duplicates
df = df.drop_duplicates()

# Ensure numeric columns are valid
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

# Remove invalid values
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert InvoiceDate safely
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df = df.dropna(subset=['InvoiceDate'])

# 5. Feature Engineering
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year
df['Hour'] = df['InvoiceDate'].dt.hour

# 6. Univariate Analysis

# Top 10 Countries by Transactions
plt.figure(figsize=(10,5))
df['Country'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Countries by Transactions")
plt.xlabel("Country")
plt.ylabel("Count")
plt.show()

# Quantity Distribution
plt.figure(figsize=(8,4))
sns.boxplot(x=df['Quantity'])
plt.title("Quantity Distribution")
plt.show()

# 7. Bivariate Analysis

# Revenue by Country
country_revenue = (
    df.groupby('Country')['Revenue']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
country_revenue.plot(kind='bar')
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Revenue")
plt.show()

# Monthly Revenue Trend
monthly_sales = df.groupby('Month')['Revenue'].sum().sort_index()

plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()

# Hourly Revenue Trend
hourly_sales = df.groupby('Hour')['Revenue'].sum().sort_index()

plt.figure(figsize=(10,5))
plt.plot(hourly_sales.index, hourly_sales.values)
plt.title("Revenue by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Revenue")
plt.show()

# 8. Multivariate Analysis (Correlation)
plt.figure(figsize=(8,5))
sns.heatmap(
    df[['Quantity', 'UnitPrice', 'Revenue']].corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.show()

# 9. Customer Analysis
top_customers = (
    df.groupby('CustomerID')['Revenue']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_customers.plot(kind='bar')
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer ID")
plt.ylabel("Revenue")
plt.show()

# 10. Key Insights
print("\nKEY INSIGHTS:")
print("1. UK contributes the highest revenue.")
print("2. Sales peak towards the end of the year.")
print("3. Most purchases happen during working hours.")
print("4. A small number of customers generate most revenue.")

print("\nEDA COMPLETED SUCCESSFULLY ✅")





# In[7]:


# ================================
# MACHINE LEARNING: CUSTOMER SEGMENTATION (ERROR-FREE)
# ================================

# Required imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Create customer-level features
customer_df = (
    df.groupby('CustomerID')
    .agg({
        'InvoiceNo': 'nunique',   # Purchase frequency
        'Quantity': 'sum',        # Total items purchased
        'Revenue': 'sum'          # Total revenue
    })
    .reset_index()
)

customer_df.columns = ['CustomerID', 'Frequency', 'TotalQuantity', 'TotalRevenue']

# 2. Handle missing or infinite values (VERY IMPORTANT)
customer_df = customer_df.replace([np.inf, -np.inf], np.nan)
customer_df = customer_df.dropna()

# 3. Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    customer_df[['Frequency', 'TotalQuantity', 'TotalRevenue']]
)

# 4. Elbow Method (safe for latest sklearn)
inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10   # fixes sklearn warning & errors
    )
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# 5. Train Final KMeans Model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customer_df['Cluster'] = kmeans.fit_predict(scaled_features)

# 6. Cluster Visualization (error-safe)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=customer_df,
    x='TotalQuantity',
    y='TotalRevenue',
    hue='Cluster'
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Total Quantity Purchased")
plt.ylabel("Total Revenue")
plt.show()


# In[8]:


# ================================
# RECOMMENDATION SYSTEM
# ================================

from sklearn.metrics.pairwise import cosine_similarity

# 1. Create Customer-Product Matrix
customer_product_matrix = df.pivot_table(
    index='CustomerID',
    columns='Description',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# 2. Compute Cosine Similarity
product_similarity = cosine_similarity(customer_product_matrix.T)

product_similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)

# 3. Recommendation Function
def recommend_products(product_name, top_n=5):
    if product_name not in product_similarity_df.columns:
        return "Product not found in database"
    
    similar_products = (
        product_similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )
    return similar_products

# 4. Example Recommendation
recommend_products("WHITE HANGING HEART T-LIGHT HOLDER")


# In[ ]:




