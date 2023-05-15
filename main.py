import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dbscan
import kmeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

def plot_clusters(data_points, labels, centroids=None, title=""):
    fig = plt.figure()
    
    if data_points.shape[1] == 2:
        plt.scatter(data_points[:, 0], data_points[:, 1], c=labels)
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
    elif data_points.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], c=labels)
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidths=3, color='r')
    else:
        print("Cannot plot data with more than 3 features")
    
    plt.title(title)

    if centroids is None and data_points.shape[1] == 2:
        # Create a color bar
        cb = plt.colorbar()
        cb.set_ticks(list(set(labels)))
        cb.set_ticklabels(list(range(len(set(labels)))))

    plt.show()

#Loading the dataset
data = pd.read_csv("marketing_campaign.csv", sep="\t")

#Data cleaning
#Remove NA values
data = data.dropna()

#Changing features to more appropriate data features
#Customer age from year of birth
data["Age"] = 2023-data["Year_Birth"]

#Total spendings on various items by adding the columns of each item
data["Spent"] = data["MntWines"]+ data["MntFruits"]
+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Customer is living with a partner or alone
#Total number of children in each household
#From those previous two columns, we can calculate the total family size
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", 
                                                    "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
data["Children"]=data["Kidhome"]+data["Teenhome"]
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

#Replacing the education column with more appropriate values (undergraduate, graduate, postgraduate)
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", 
                                             "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#Dropping extra features we don't need
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)

#Delete outliers by setting a cap on Age and income. 
data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]

# Encode categorical values as integers for dbscan and kmeans
cat_columns = [col for col in data.columns if data[col].dtype == 'object']
label_encoders = {}
for col in cat_columns:
    label_encoders[col] = LabelEncoder()
# Encode categorical values in each column
for col, enc in label_encoders.items():
    data[col] = enc.fit_transform(data[col])

# Standardize the data with StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaled_ds = pd.DataFrame(scaler.transform(data),columns= data.columns )
aux = scaled_ds.values




# Use PCA (Principal Component Analysis) to reduce the dimensionality of the data
pca = PCA(n_components=2)
data_pca = pca.fit_transform(aux)

# Run K-means clustering with K=3 and maximum 100 iterations
centroids, closest = kmeans.k_means(data_pca, num_clusters=3, max_iterations=100)

# Set the parameters for DBSCAN
eps = 5 # radius of the neighborhood
min_samples = 100 # minimum number of points to form a dense region

labels = dbscan.dbscanFunction(data_pca, eps, min_samples)

# Plotting K-means results
plot_clusters(data_pca, closest, centroids, title="K-means results")
# Plotting DBSCAN results
plot_clusters(data_pca, labels, title="DBSCAN results")


# Adding the cluster labels to the original dataset
data['KMeans_Cluster'] = closest
data['DBSCAN_Cluster'] = labels


# Plotting the clusters in the original feature space (income / spent)
x = data['Spent']
y = data['Income']
cluster_labels = data['KMeans_Cluster']

plt.scatter(x, y, c=cluster_labels, cmap='viridis')
plt.xlabel('Spent')
plt.ylabel('Income')
plt.title('Scatter Plot: Spent vs Income (K-Means)')
plt.colorbar(label='Cluster Labels')
plt.show()

cluster_labels = data['DBSCAN_Cluster']

plt.scatter(x, y, c=cluster_labels, cmap='plasma')
plt.xlabel('Spent')
plt.ylabel('Income')
plt.title('Scatter Plot: Spent vs Income (DBSCAN)')
plt.colorbar(label='Cluster Labels')
plt.show()


# Number of instances in each cluster

values, counts = np.unique(closest, return_counts=True)
for value, count in zip(values, counts):
    print("K Means"f"{value}: {count}")

values, counts = np.unique(labels, return_counts=True)
for value, count in zip(values, counts):
    print("DBSCAN"f"{value}: {count}")