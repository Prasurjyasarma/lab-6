import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("iris.csv")

# Identify categorical and numerical variables
cat = list(df.select_dtypes(include=['object']).columns)
num = list(df.select_dtypes(exclude=['object']).columns)

# Print categorical and numerical variables
print(f'Categorical variables: {cat}')
print(f'Numerical variables: {num}')

# Create a pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of the dataset", y=1.02)
plt.show()

# Check if the last column is categorical and plot accordingly
if df.iloc[:, -1].dtype == 'object':
    sns.countplot(x=df.columns[-1], data=df, hue=df.columns[-1])
    plt.title('Bar chart of categorical column')
    plt.show()
else:
    print("No categorical value")
