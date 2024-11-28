# %% [markdown]
# # Programming Assignment 3

# %% [markdown]
# In this assignment, you will implement k-means and hierarchical clustering. You can use any library functions to implement the tasks.

# %% [markdown]
# ## The Wine Dataset

# %% [markdown]
# ### Dataset Description:
# 
# These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
# 
# The attributes are 
# 1) Alcohol  
# 2) Malic acid  
# 3) Ash  
# 4) Alcalinity of ash   
# 5) Magnesium  
# 6) Total phenols  
# 7) Flavanoids  
# 8) Nonflavanoid phenols  
# 9) Proanthocyanins  
# 10)Color intensity  
# 11)Hue  
# 12)OD280/OD315 of diluted wines  
# 13)Proline   
# 
# The first five instances of the dataset are shown below:

# %%
import pandas as pd

data=pd.read_csv('wine.csv')
data.head(5)

# %% [markdown]
# All columns except for the first one contain different features. The first column contains the class label (type of wine). DO NOT use data from the first column (i.e., the class labels) as feature.

# %% [markdown]
# ## Task 1: k-means Clustering (50 points)
# 
# The basic K-means clustering works as follows:  
# 
#     1.Initialize ‘K’, number of clusters to be created.  
#     2.Randomly assign K centroid points.  
#     3.Assign each data point to its nearest centroid to create K clusters.  
#     4.Re-calculate the centroids using the newly created clusters.  
#     5.Repeat steps 3 and 4 until the centroid gets fixed.  
# 
# Apply k-means clustering on the wine dataset using euclidean distance as the distance metric and print the SSE (sum of squared errors) values for k = 2-8. (print the error values in %.2f format)
# 
# Example Output:  
# For k = 2 After 15 iterations: Error =     
# For k = 3 After 15 iterations: Error =   
# For k = 4 After 15 iterations: Error =   
# For k = 5 After 15 iterations: Error =   
# For k = 6 After 15 iterations: Error =  
# For k = 7 After 15 iterations: Error =   
# For k = 8 After 15 iterations: Error =  
# 
# Display the SSE Error vs k plot and using elbow method, choose a good value for k.

# %% [markdown]
# ## Task 2: Agglomerative Hierarchical Clustering (50 points)
# 
# The basic agglomerative hierarchical clustering works as follows:
# 
#     1. Initialize the proximity matrix 
#     2. Let each data point be a cluster 
#     3. Repeat the following steps until there is only one cluster left:
#         a. Find the two closest clusters in the cluster list and merge them into a single cluster
#         b. Update the cluster distance matrix to reflect the new cluster's distances to the other clusters
#         c. Remove the two clusters that were just merged and add the new single cluster to the cluster list.
#     4. Return the final single cluster, which is the hierarchy of clusters.
#     
# a) Perform Single link hierarchical clustering on the wine dataset using euclidean distance as the distance metric. Show the dendrogram and calculate the silhouette coefficient for k = 3.
# 
# b) Perform Complete link hierarchical clustering on the wine dataset using euclidean distance as the distance metric. Show the dendrogram and calculate the silhouette coefficient for k = 3.

# %% [markdown]
# ### Submission Guidelines:
#         
# Submit through Canvas your source code in a single .ipynb file.   
# The name of the .ipynb file should be YourStudentID.ipynb. (For example: 1001234567.ipynb)  
# You don't need to attach the dataset with your submission.  


