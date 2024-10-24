import pandas as pd

# Loading dataset
data = pd.read_csv(r'C:\\Users\\Rishav Roshan\\OneDrive\\Desktop\\ML Project\\Task_2\\Mall_Customers.csv')

# Checking the first few rows
data.head()

# Checking for missing values
print(data.isnull().sum())

# Drops irrelevant columns (like CustomerID if it exists)
data = data.drop(columns=['CustomerID'])

# Optional: Rename columns if necessary for clarity
data.columns = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Converts Gender to numerical (optional: OneHotEncoding)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Use the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Add cluster labels to the original data
data['Cluster'] = y_kmeans

# Save the results for further analysis
data.to_csv('Customer_Segmentation_Results.csv', index=False)

