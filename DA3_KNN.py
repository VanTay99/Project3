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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2,random_state=41)
# ,stratify=iris_y)  
#  120 / 30

clf = neighbors.KNeighborsClassifier(n_neighbors = 1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

distance, nn = clf.kneighbors(X_test.iloc[15].to_numpy().reshape(1,-1),5)
# 119 ,113,83
print("Wrong predict",iris_y[119],iris_y[113],iris_y[83])
print("Nearest neighbors\n",distance,'\n',nn,'\n',y_train.iloc[nn[0]])
print("\nPrint results for test data points:")
print("\nPredicted labels:\n ", y_pred[0:5])
print("\nTrue labels     :\n ", y_test[0:5])

from sklearn.metrics import classification_report, confusion_matrix,precision_score,recall_score,f1_score

print("\nPrecision score:",precision_score(y_test, y_pred,average='macro'))
print('recall:',recall_score(y_test, y_pred,average='macro'))
print('f1-score:',f1_score(y_test, y_pred,average='macro'))

labels=['setosa', 'versicolor' ,'virginica']
print("\nconfusion_matrix:\n",pd.DataFrame(confusion_matrix(y_test, y_pred,labels=labels), index=labels, columns=labels))
print("\nclassification_report:\n",classification_report(y_test, y_pred))


from sklearn import metrics
k_range = list(range(1, 26))  
scores={}
list_score_uniform=[]
for k in k_range:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k,weights='uniform')
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    precision = precision_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    f1 = f1_score(y_test, y_pred,average='macro')
    scores[k] = {
      'precision':round(precision,5),
      'recall':round(recall,5),
      'f1':round(f1,5)
    }
    list_score_uniform.append(f1)

print('scores uniform:\n')
print("{:<8} {:<15} {:<8} {:<8}".format('Key','Precision','Recall','f1'))
for k, v in scores.items():
  print("{:<8} {:<15} {:<8} {:<8}".format(k,v['precision'],v['recall'],v['f1']))

from sklearn import metrics
k_range = list(range(1, 26))  
scores={}
list_score_distance=[]
for k in k_range:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k,weights='distance')
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    precision = precision_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    f1 = f1_score(y_test, y_pred,average='macro')
    scores[k] = {
      'precision':round(precision,5),
      'recall':round(recall,5),
      'f1':round(f1,5)
    }
    list_score_distance.append(f1)

print('scores distance:\n')
print("{:<8} {:<15} {:<8} {:<8}".format('Key','Precision','Recall','f1'))
for k, v in scores.items():
  print("{:<8} {:<15} {:<8} {:<8}".format(k,v['precision'],v['recall'],v['f1']))

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
# plt.show()
