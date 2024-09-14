import pandas as pd
import numpy as np
import  csv
from sklearn.preprocessing import StandardScaler
from  sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

try :
    data1=pd.read_csv("test.csv", sep=',')
except Exception as e:
    print(f"error lors de l'ouverture du fichier :{e}")
print(data1.head())
print(data1.columns)
data1 = pd.DataFrame(data=data1)

#exclure  id
x= data1.drop('id', axis=1)

#puisque les données les données ont des échelles différentes :
# on va les standariser:
Scaler = StandardScaler()
data_scaled = Scaler.fit_transform(x)
print(data_scaled)
kmeans = KMeans(n_clusters=3,random_state=42 )
#random_state pour fixer les premiers centres à chaque exécution
data1['price_range']=kmeans.fit_predict(data_scaled)
print(data1)
clusters = kmeans.fit_predict(data_scaled)
print(clusters)

score = silhouette_score(data_scaled, clusters)
print(f"Silhouette Score: {score:.4f}")

# Calcul des moyennes des caractéristiques pour chaque cluster (price_range)
x= data1.drop('id', axis=1)
cluster_means =x.groupby('price_range').mean()


# Calculer la moyenne générale de chaque cluster (moyenne de toutes les colonnes)
cluster_means['global_mean'] = cluster_means.mean(axis=1)
print(cluster_means)
data1['price_range'] = data1['price_range'].replace({0: 'basse_gamme', 1:'haute_gamme' , 2:'moyenne_gamme'})
print(data1)
print(data1[['id','price_range']])
#visualisation
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA
import seaborn as sns
#réductions des données :
X = data1.drop(['price_range', 'id'], axis=1)
pca= PCA(n_components=2)
x_pca=pca.fit_transform(x)
pca_df = pd.DataFrame(x_pca, columns=['PC1', 'PC2'])
pca_df['price_range'] = data1['price_range']
print(pca_df)
palette1 = {'basse_gamme': 'blue' ,'moyenne_gamme':'green' , 'haute_gamme':'red'}
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='price_range', data=pca_df, palette=palette1, alpha=0.7)
plt.title('Visualisation des Clusters K-Means avec PCA ', fontsize=14)
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')

plt.grid(True)
plt.show()
# Réduire les dimensions avec t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data_scaled)

# Créer un DataFrame pour visualiser
tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['price_range'] = data1['price_range']
print(tsne_df)

# Visualiser
palette={'basse_gamme': 'blue' ,'moyenne_gamme':'green' , 'haute_gamme':'red'}
plt.figure(figsize=(10, 8))
sns.scatterplot(x='t-SNE 1', y='t-SNE 2', hue='price_range', data=tsne_df, palette=palette)
plt.title('Visualisation des Clusters avec t-SNE', fontsize=14)
plt.grid(True)
plt.show()

df_bas = data1[data1['price_range'] == 'basse_gamme']
df_moyen=data1[data1['price_range']=='moyenne_gamme']
df_haut=data1[data1['price_range']=='haute_gamme']


print(df_bas, df_moyen,df_haut ,sep='\n')
df_bas.to_csv('basse_gamme.csv', index=False)
df_moyen.to_csv('moyenne_gamme.csv', index=False)
df_haut.to_csv('haute_gamme.csv', index=False)



