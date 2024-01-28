# Kütüphaneleri İçeri Aktarmak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri Setini İçeri Aktarmak
dataset = pd.read_csv('C:/Users/holig/OneDrive/Belgeler/Vscode/opencv/Hakan_data_mine_odev/stocks.csv')

# Bağımsız değişkenler (X) ve Hedef değişken (y)
X = dataset.iloc[:, 2:].values
y = dataset["Close"].values

# Veri setini eğitim ve test kümesi olarak bölümlendirmek
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Öznitelik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)

# Karar Ağacı Sınıflandırma Modelini Eğitim Kümesinde Eğitmek
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Yeni bir sonuç tahmin etmek
new_data = np.array([[30, 140000, 0, 0, 0, 0]])
new_data_scaled = sc.transform(new_data)
prediction = regressor.predict(new_data_scaled)
print(prediction)

# Test kümesi sonuçlarını tahmin etmek
y_pred = regressor.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Hata metriklerini elde etmek
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R^2:", r2)
