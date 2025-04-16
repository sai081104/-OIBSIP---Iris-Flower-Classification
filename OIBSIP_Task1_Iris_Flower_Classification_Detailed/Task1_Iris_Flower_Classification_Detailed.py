# 🌸 Task 1: Iris Flower Classification - OIBSIP Data Science Internship

# This project uses a machine learning model to classify Iris flowers into
# three species based on their petal and sepal dimensions.

# ✅ Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ Step 1: Load the dataset
# The dataset includes 150 samples with 4 features: 
# sepal length, sepal width, petal length, petal width
# and a label: species (Setosa, Versicolor, Virginica)

df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv")

# ✅ Step 2: Display first few rows
print("Sample data:")
print(df.head())

# ✅ Step 3: Define features (X) and labels (y)
X = df.drop("species", axis=1)  # input features
y = df["species"]               # output labels

# ✅ Step 4: Split data into training and test sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 5: Train the model
# Using Random Forest Classifier for better accuracy
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Step 6: Make predictions on the test set
predictions = model.predict(X_test)

# ✅ Step 7: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

# ✅ Step 8: View predictions (optional)
print("\nPredicted labels:")
print(predictions)

print("\nActual labels:")
print(list(y_test.values))