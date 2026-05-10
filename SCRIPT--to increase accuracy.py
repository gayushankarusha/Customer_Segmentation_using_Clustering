# 📌 Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 📌 Step 2: Load dataset
df = pd.read_csv("C:\\Users\\USER\Desktop\customer segmentation using kmeans clustering\customer_segmentation _dataset.csv")
  # replace with your dataset file

# Features and target
X = df.drop("Cluster", axis=1)   # independent variables
y = df["Cluster"]                # target variable

# 📌 Step 3: Handle imbalance (upsample minority classes)
df_majority = df[df.Cluster == 0]
df_minority = df[df.Cluster == 1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
X = df_balanced.drop("Cluster", axis=1)
y = df_balanced["Cluster"]

# 📌 Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 📌 Step 5: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Step 6: Compare multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))

# 📌 Step 7: Hyperparameter tuning (Gradient Boosting example)
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print("\nBest Gradient Boosting Params:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)
