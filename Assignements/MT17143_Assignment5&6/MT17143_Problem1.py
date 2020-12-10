#MT17143 Akshita Sawhney
#Problem 1 RNA Sequencing

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math
montpick= open("Read_count_rattus.txt",'r')
matrix = []
read_file = montpick.readlines()
for line in read_file:                                  #file is extracted in a 2D matrix
    row = []
    list_of_words = line.split()
    for i in range(1, len(list_of_words)):
        row.append(int(list_of_words[i]))
    matrix.append(row)

#Normalization
trc = 0                         # total read count is calculated
for l in matrix:
    for el in l:
        trc+=el

sum=0
count=0
# print(len(matrix[1]))
for i in range(len(matrix[0])):                 # Sum of each column is calculated
    column_sum = 0
    for l in matrix:
        column_sum += l[i]
    sum+=column_sum
sum=sum/len(matrix[0])

for l in matrix:                                  #Each readcount value is divided by the total read count
    for i in range(len(l)):
        div = float(l[i]/trc)
        l[i]=div


for l in matrix:                                #Each readcount value is then multiplied by the sum of columns
    for i in range(len(l)):
        l[i]= float(l[i] * sum)

#Log Transform
for l in matrix:
    for i in range(len(l)):
        l[i]=math.log(1+l[i],2)

# print(matrix)
# print("hi")
input_matrix = np.array(matrix)
# print(M)
#The actual data matrix is extracted from the phenodata which acts as the true data.
phenotype = []
phenodata = open("Phenotype.txt",'r')
lines= phenodata.readlines()
for l in lines:
    phen = l.split()
    phenotype.append(int(phen[0]))
# phenotype1 = phenotype[1:]

true_matrix= np.array(phenotype)

#Input Data is split into Train and Test set with test size to be 33%
X_train, X_test, y_train, y_test  = train_test_split(np.transpose(input_matrix),true_matrix,test_size=0.33)
#Kmeans Clustering is performed
kmeans=KMeans(n_clusters=2, random_state= 0).fit(X_train)
kmean_prediction = kmeans.predict(X_test)    #Test data is passed to check the results.
print(accuracy_score(y_test,kmean_prediction)*100)          # Accuracy of the predicted output with true data is taken out.

X_train, X_test, y_train, y_test  = train_test_split(np.transpose(input_matrix),true_matrix,test_size=0.33)
#MiniBatchKmeans clustering is performed
Minibatchkmeans = MiniBatchKMeans(n_clusters=2, random_state= 0).fit(X_train)
minibatchkmean_prediction = Minibatchkmeans.predict(X_test)
print(accuracy_score(y_test,minibatchkmean_prediction)*100)

#Principle Component Analysis is performed to reduce the input data to 2Dimensional data.
pca = PCA(n_components=2).fit_transform(np.transpose(input_matrix))
# pca_fit = pca.fit(np.transpose(input_matrix))
y_trans = np.transpose(true_matrix)
plt.scatter(pca[:, 0], pca[:, 1], y_trans.shape[0], c = y_trans)    #Scatter is used to visualize the graph
plt.show()