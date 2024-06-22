import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("D:/Prashant/house.csv")

# Separate features and target variable
X = df.drop(columns=['price'], axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Plot actual vs predicted values
sns.regplot(x=y_test, y=y_pred, scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted Values')
plt.show()

# Prompt the user for input
size = float(input("Enter size of house in sq. ft: "))
age = float(input("Enter the age of house in years: "))

# Predict the price for the user-provided input
user_input = np.array([[size, age]])
predicted_house = model.predict(user_input)

print(f"Predicted price for a house with size {size} sq. ft and age {age} years: Rs. {predicted_house[0]:.2f} lakhs")
