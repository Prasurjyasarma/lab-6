import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("D:/Prashant/weather.csv")

data_encoded = pd.get_dummies(df, columns=['outlook', 'Temp', 'Humidity', 'windy'])


X = data_encoded.drop('Play Tennis', axis=1) 
y = data_encoded['Play Tennis']




dtc = DecisionTreeClassifier()
dtc.fit(X, y)


plt.figure(figsize=(20, 10))
plot_tree(dtc, feature_names=X.columns, class_names=y.unique(y), filled=True)
plt.show()