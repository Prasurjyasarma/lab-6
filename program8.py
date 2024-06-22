import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C://Prashant//house.csv")
x = df.drop(columns=["price"], axis=1)
y = df["price"]

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

sns.regplot(x=y, y=y_pred, scatter_kws={"s": 10}, line_kws={"color": "red"})
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted Values")
plt.show()

size = float(input("Enter size of house in sq. ft.: "))
age = float(input("Enter the age of house in years: "))
user_input = np.array([[size, age]])
predicted_house = model.predict(user_input)

print("Predicted price for a house with size {0} sq. ft. and age {1} years is: Rs. {2} lakhs".format(size, age, predicted_house[0]))
