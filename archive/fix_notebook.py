import json

notebook_path = 'Student_Grade_Prediction.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Remove colab drive mount
nb['cells'][1]['source'] = [
    "# Removed Google Colab drive mount since we are running locally\n"
]

# Cell 2: Imports and Data Loading
nb['cells'][2]['source'] = [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "from sklearn.utils import shuffle\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Read the dataset directly from the local directory.\n",
    "mat_data = pd.read_csv('student-mat.csv')\n",
    "mat_data.head()\n"
]

# Cell 20: Categorical Encoding
nb['cells'][20]['source'] = [
    "# Convert categorical variables into dummy/indicator variables (One-Hot Encoding)\n",
    "# We use drop_first=True to avoid the 'dummy variable trap' (perfect multicollinearity) which can mess up Linear Regression.\n",
    "# pd.get_dummies is preferred over .iloc with LabelEncoder because it avoids SettingWithCopyWarnings and prevents the model from assuming an artificial ordering in nominal categorical variables.\n",
    "mat_data = pd.get_dummies(mat_data, drop_first=True)\n",
    "mat_data.head()\n"
]

# Cell 24: Dropping columns
nb['cells'][24]['source'] = [
    "# Previously, G1 and G2 were dropped here, which caused the R^2 to plummet because they are the strongest predictors of G3.\n",
    "# We remove the drop step to keep G1 and G2 intact.\n"
]

# Cell 25: Feature selection logic
nb['cells'][25]['source'] = [
    "# Calculate the absolute correlation of all features with the target variable G3\n",
    "correlations = mat_data.corr().abs()['G3'].sort_values(ascending=False)\n",
    "\n",
    "# Select the top correlated features, but ensure we explicitly include 'G1', 'G2', 'studytime', and 'absences'\n",
    "top_features = correlations[:10].index.tolist() # Take the top 10 most correlated\n",
    "\n",
    "# Ensure required features are present (adding them if they aren't in the top 10)\n",
    "required_features = ['G1', 'G2', 'studytime', 'absences', 'G3']\n",
    "for feature in required_features:\n",
    "    if feature not in top_features and feature in mat_data.columns:\n",
    "        top_features.append(feature)\n",
    "\n",
    "print(f'Selected Features: {top_features}')\n"
]

# Cell 26: Applying feature selection
nb['cells'][26]['source'] = [
    "# Filter the dataset to keep only the selected features for our model\n",
    "mat_data = mat_data[top_features]\n",
    "mat_data.head()\n"
]

# Cell 30: Model Training and Evaluation
nb['cells'][30]['source'] = [
    "# Separate the features (X) from the target variable (y)\n",
    "X_train = X_train.drop('G3', axis='columns')\n",
    "X_test = X_test.drop('G3', axis='columns')\n",
    "\n",
    "# Initialize and train the Linear Regression model\n",
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions on the test set\n",
    "predictions = lr.predict(X_test)\n",
    "\n",
    "# Calculate evaluation metrics to understand model performance\n",
    "mae = mean_absolute_error(y_test, predictions)\n",
    "mse = mean_squared_error(y_test, predictions)\n",
    "rmse = np.sqrt(mse)\n",
    "r2 = r2_score(y_test, predictions)\n",
    "\n",
    "print('--- Model Evaluation Metrics ---')\n",
    "print(f'Mean Absolute Error (MAE): {mae:.4f}')\n",
    "print(f'Mean Squared Error (MSE): {mse:.4f}')\n",
    "print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')\n",
    "print(f'R-squared (R2): {r2:.4f}')\n"
]

# Cell 31: Sample Predictions vs Actual
nb['cells'][31]['source'] = [
    "# Show a comparison of actual vs predicted grades for the first few samples\n",
    "comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})\n",
    "comparison.head(10)\n"
]

# Cell 32: Plotting Actual vs Predicted
nb['cells'][32]['source'] = [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Set up the visual aesthetics\n",
    "sns.set_theme(style='whitegrid')\n",
    "plt.figure(figsize=(10, 6))\n",
    "\n",
    "# Create a scatter plot of Actual vs Predicted grades\n",
    "sns.scatterplot(x=y_test, y=predictions, alpha=0.7, color='blue', edgecolor='k', s=80)\n",
    "\n",
    "# Plot the line of perfect prediction (y = x)\n",
    "max_val = max(max(y_test), max(predictions))\n",
    "min_val = min(min(y_test), min(predictions))\n",
    "plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')\n",
    "\n",
    "plt.title('Actual vs Predicted Grades (G3)', fontsize=16)\n",
    "plt.xlabel('Actual Grade (G3)', fontsize=14)\n",
    "plt.ylabel('Predicted Grade', fontsize=14)\n",
    "plt.legend()\n",
    "plt.show()\n"
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
