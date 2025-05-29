import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load Dataset
df = pd.read_csv('data.csv', encoding='windows-1252') 

# 2. Data Exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Handling missing data example
df.dropna(inplace=True)

# 3. Feature Engineering (Example for RFM)
# Assuming you have columns: CustomerID, InvoiceDate, InvoiceNo, Amount
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

df['Amount'] = df['Quantity'] * df['UnitPrice']


rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'Amount': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Amount': 'Monetary'})

print(rfm.head())

# 4. Scaling Features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# 5. Find Optimal Number of Clusters (Elbow Method)
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

plt.plot(range(2, 11), sse)
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()

# 6. Fit K-Means with chosen clusters (e.g., k=4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 7. Cluster Visualization
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='Set1')
plt.title('Customer Segments based on Recency and Monetary')
plt.show()

# 8. Cluster Profiles
cluster_profiles = rfm.groupby('Cluster').mean()
print(cluster_profiles)

# 9. Recommendations
print("""
Insights and Recommendations:
- Cluster 0: High-value and frequent customers — focus on loyalty programs.
- Cluster 1: New or recent customers with low purchase frequency — engage with introductory offers.
- Cluster 2: Customers with high recency but low monetary value — consider re-engagement campaigns.
- Cluster 3: Low activity and low spending — may be less profitable, consider cost-effective retention.
""")
