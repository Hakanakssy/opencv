
# Karar Ağacı Sınıflandırma
## Kütüphaneleri İçeri Aktarmak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Veri Setini İçeri Aktarmak
dataset = pd.read_csv('SmokeBan.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Veri setini eğitim ve test kümesi olarak bölümlendirmek
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

## Öznitelik ölçeklendirme
drop_columns = ['age']
X_train = X_train.drop(drop_columns, axis=1)
X_test = X_test.drop(drop_columns, axis=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)

## Karar Ağacı Sınıflandırma Modelini Eğitim Kümesinde Eğitmek
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

## Yeni bir sonuç tahmin etmek
print(classifier.predict(sc.transform([[30,140000]])))

## Test kümesi sonuçlarını tahmin etmek
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

## Hata matrisini elde etmek
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
