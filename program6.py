import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('D:/Prashant/train.csv')

# Display the first three rows of the dataframe
print(df.head(3))

# Display the number of missing values per column
print(df.isnull().sum())

# Drop the 'Cabin' column
df = df.drop(columns='Cabin', axis=1)

# Fill missing values in 'Age' with the mean value of 'Age'
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing values in 'Embarked' with the mode value of 'Embarked'
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Check for any remaining missing values
print(df.isnull().sum().sum())

# Drop unnecessary columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1)

# Encode categorical variables
le = LabelEncoder()
df['Embarked'] = le.fit_transform(df['Embarked'])
df['Sex'] = le.fit_transform(df['Sex'])

# Display dataframe info
print(df.info())

# Separate features and target variable
X = df.drop(columns=['Survived'], axis=1)
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Display the standardized training set
print(X_train)
