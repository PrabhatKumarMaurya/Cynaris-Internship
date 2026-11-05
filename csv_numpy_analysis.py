import pandas as pd
import numpy as np

# Read CSV dataset
df = pd.read_csv("iris.csv")

# Convert columns into NumPy arrays
sepal_length = df["sepal length (cm)"].to_numpy()
petal_length = df["petal length (cm)"].to_numpy()

# Calculate statistics
mean_value = np.mean(sepal_length)
std_value = np.std(sepal_length)
correlation = np.corrcoef(sepal_length, petal_length)[0, 1]

# Display results
print("Dataset Shape:", df.shape)
print("Mean Sepal Length:", mean_value)
print("Standard Deviation:", std_value)
print("Correlation:", correlation)