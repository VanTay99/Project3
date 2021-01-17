import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
import pandas as pd

iris = datasets.load_iris()
df=pd.DataFrame(iris['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])
df["Species"]=iris["target"]
df["Species"]=df["Species"].apply(lambda x: iris["target_names"][x])
print(df.head())
print(df.describe())
iris_X = df[["Petal length","Petal Width","Sepal Length","Sepal Width"]]
iris_y = df["Species"]
print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))

# print(iris_X,iris_y)
# X0 = iris_X[iris_y == 0,:]
# print ('\nSamples from class 0:\n', X0[:5,:])

# X1 = iris_X[iris_y == 1,:]
# print ('\nSamples from class 1:\n', X1[:5,:])

# X2 = iris_X[iris_y == 2,:]
# print ('\nSamples from class 2:\n', X2[:5,:])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3,random_state=25,stratify=iris_y)

clf = neighbors.KNeighborsClassifier(n_neighbors = 4, p = 2)
# clf = neighbors.KNeighborsClassifier(n_neighbors = 5, p = 2,weights='distance')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# print('aa',clf.kneighbors([X_test[0]]))
# from sklearn.neighbors import NearestNeighbors
# neigh = NearestNeighbors(n_neighbors=10)
# # X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13]).reshape(-1,1)
# # print(X)
# neigh.fit(X_train)
# # neigh.fit(X)
# print(neigh.kneighbors([X_test[0]]))


from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores,sum(scores)/5)
print("Print results for test data points:")
print("Predicted labels: ", y_pred)
print("True labels     : ", y_test)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix,precision_score,recall_score

print("Pricesion score:",precision_score(y_test, y_pred,average='macro'))
labels=['setosa', 'versicolor' ,'virginica']
print("\nconfusion_matrix:\n",pd.DataFrame(confusion_matrix(y_test, y_pred,labels=labels), index=labels, columns=labels))
print("\nclassification_report:\n",classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 50))  
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=k_range, weights=weight_options)
knn = neighbors.KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X_train,y_train)

print(knn_cv.best_params_,knn_cv.best_score_)

from sklearn import metrics
k_range = list(range(1, 30))  
# weight_options = ['uniform', 'distance']
scores={}
list_score_uniform=[]
for k in k_range:
  # for weight in weight_options:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k,weights='uniform')
    # ,weights=weight)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    # scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    # scores = precision_score(y_test, y_pred,average='macro')
    # precision_score(y_test, y_pred)
    # print(scores,sum(scores)/5)
    # list_score_uniform.append(scores)
    list_score_uniform.append(sum(scores)/5)
    # list_score_uniform.append(metrics.accuracy_score(y_test,y_pred)) 

from sklearn import metrics
k_range = list(range(1, 30))  
# weight_options = ['uniform', 'distance']
scores={}
list_score_distance=[]
for k in k_range:
  # for weight in weight_options:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k,weights='distance')
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    # print(scores,sum(scores)/5)
    # scores = precision_score(y_test, y_pred,average='macro')
    # precision_score(y_test, y_pred)
    # list_score_distance.append(scores)
    list_score_distance.append(sum(scores)/5)
    # y_pred = knn.predict(X_test)
    # scores[k] = metrics.accuracy_score(y_test,y_pred)
    # list_score_distance.append(metrics.accuracy_score(y_test,y_pred)) 

# print(list_score)
# # print(scores)
fig = plt.figure() 
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(k_range,list_score_uniform)
ax1.set_xlabel("Value K for KNN")
ax1.set_ylabel("Testing Precision")
ax1.set_title("Weight: Uniform",loc='right')

ax2.plot(k_range,list_score_distance)
ax2.set_xlabel("Value K for KNN")
ax2.set_ylabel("Testing Precision")
ax2.set_title("Weight: Distance",loc='right')
plt.show()
