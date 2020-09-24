import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
# print(data.head())

X = data[[
    'buying',
    'maint',
    'safety',
]].values

Y = data[['class']]

# print(X,Y)

Encode = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Encode.fit_transform(X[:, i])
# print(X)

label_mapping = {
    'unacc' : 0,
    'acc' : 1,
    'good' : 2,
    'vgood' : 3
}

Y['class'] = Y['class'].map(label_mapping)
Y = np.array(Y)
# print(Y)

knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = "uniform")

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

knn.fit(X_train, Y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(Y_test,prediction)
print("Prediction: ", prediction)
print("Accuracy: ", accuracy)

print("actual value:",Y[20])
print("predicted value:",knn.predict(X)[20])