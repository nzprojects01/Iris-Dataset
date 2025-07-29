import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the dataset
df = pd.read_csv("D:/iris.csv")

# Understand data
print(df.head())
print(df.dtypes)

# View Missing Data
print(df.isnull().sum())
