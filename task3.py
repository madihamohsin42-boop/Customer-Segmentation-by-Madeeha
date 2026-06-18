import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. DATA LOAD
df = pd.read_csv('mall-customers.csv')
print("Data loaded. Shape:", df.shape)
print(df.head(), "\n")

# 2. SCALE - صرف Income اور Spending Score لیں گے
X = df[['age', 'price']] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. COMPRESS - PCA 2D 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance retained by PCA: {pca.explained_variance_ratio_.sum():.2%}\n")

# 4. ELBOW METHOD - Best K finding
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method - Best K')
plt.xlabel('Number of Clusters K')
plt.ylabel('WCSS')
plt.savefig('elbow_plot.png')
plt.show()

# 5. CLUSTER - K=5 best data 
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
df['Cluster'] = clusters

# 6. SILHOUETTE SCORE
score = silhouette_score(X_pca, clusters)
print(f"Silhouette Score for K={k}: {score:.3f}\n")

# 7. PLOT CUSTOMERS
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', s=60, alpha=0.7)
plt.title('Customer Segments - 3 Personas')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.savefig('customer_segments.png')
plt.show()

# 8. TRANSLATE - Persona Matrix بناؤ
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids_scaled))
persona_df = pd.DataFrame(centroids_original, columns=['Income (k$)', 'Spending Score'])
print("Persona Matrix - Cluster Centers:")
print(persona_df.round(2))