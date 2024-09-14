import pandas as pd
import numpy as np
import  csv
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
try :
    data=pd.read_csv("test.csv", sep=',')
except Exception as e:
    print(f"error lors de l'ouverture du fichier :{e}")
print(data.head())
print(data.columns)
data = pd.DataFrame(data=data)
#exclure  id
x= data.drop('id', axis=1)

#puisque les données les données ont des échelles différentes :
# on va les standariser:
Scaler = StandardScaler()
data_scaled = Scaler.fit_transform(x)
print(data_scaled)
gmm = GaussianMixture(n_components=3, random_state=42)
#random_state pour fixer les premiers centres à chaque exécution
data['price_range']=gmm.fit_predict(data_scaled)
print(data)

# Calcul des moyennes des caractéristiques pour chaque cluster (price_range)
x= data.drop('id', axis=1)
cluster_means =x.groupby('price_range').mean()


# Calculer la moyenne générale de chaque cluster (moyenne de toutes les colonnes)
cluster_means['global_mean'] = cluster_means.mean(axis=1)
print(cluster_means)
data['price_range'] = data['price_range'].replace({0: 'bas', 1:'haut' , 2:'moyen'})
print(data)
print(data[['id','price_range']])
clusters = gmm.fit_predict(data_scaled)
print(clusters)

score = silhouette_score(data_scaled, clusters)
print(f"Silhouette Score: {score:.4f}")
#visualisation
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA
import seaborn as sns
#réductions des données :
X = data.drop(['price_range', 'id'], axis=1)
pca= PCA(n_components=2)
x_pca=pca.fit_transform(x)
pca_df = pd.DataFrame(x_pca, columns=['PC1', 'PC2'])
pca_df['price_range'] = data['price_range']
print(pca_df)
palette1 = {'bas': 'blue' ,'moyen':'green' , 'haut':'red'}
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='price_range', data=pca_df, palette=palette1, alpha=0.7)
plt.title('Visualisation des Clusters K-Means avec PCA classes', fontsize=14)
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')

plt.grid(True)
plt.show()
from sklearn.manifold import TSNE

# Réduire les dimensions avec t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data_scaled)

# Créer un DataFrame pour visualiser
tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['price_range'] = data['price_range']
print(tsne_df)

# Visualiser
palette={'bas': 'blue' ,'moyen':'green' , 'haut':'red'}
plt.figure(figsize=(10, 8))
sns.scatterplot(x='t-SNE 1', y='t-SNE 2', hue='price_range', data=tsne_df, palette=palette)
plt.title('Visualisation des Clusters avec t-SNE', fontsize=14)
plt.grid(True)
plt.show()
