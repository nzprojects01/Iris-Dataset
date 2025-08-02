import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("D:/iris.csv")

# Understand data
print(df.head())
print(df.dtypes)

# View Missing Data
print(df.isnull().sum())

# Visualize data
# Scatter Plot
plt.scatter(df['sepal.length'], df['petal.length'])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Sepal Length Vs Petal Length")
plt.show()

# Countplot
sns.countplot(data = df, x = 'variety')
plt.show()

# Set up PairGrid
g = sns.PairGrid(data = df, hue="variety", corner=True)  # Only view the lower triangle

# Map the plots
g.map_lower(sns.scatterplot)
g.map_diag(sns.kdeplot)  # or use sns.histplot for histogram

# Add legend
g.add_legend()

plt.show()

# Split the dataset into train and test set for logistic regression
X = df.drop('variety', axis=1)  # features
y = df['variety']

# 80 percent training and 20 percent testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17
)

# Train Model
# Create logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
