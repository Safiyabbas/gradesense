import json

notebook_path = 'Student_Grade_Prediction.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 20: pd.get_dummies with dtype=int
nb['cells'][20]['source'] = [
    "# Convert categorical variables into dummy/indicator variables (One-Hot Encoding)\n",
    "# We use drop_first=True to avoid the 'dummy variable trap' (perfect multicollinearity) which can mess up Linear Regression.\n",
    "# Setting dtype=int ensures compatibility with all pandas versions for correlation.\n",
    "# pd.get_dummies is preferred over .iloc with LabelEncoder because it avoids SettingWithCopyWarnings and prevents the model from assuming an artificial ordering in nominal categorical variables.\n",
    "mat_data = pd.get_dummies(mat_data, drop_first=True, dtype=int)\n",
    "mat_data.head()\n"
]

# Cell 23
nb['cells'][23]['source'] = [
    "mat_data.corr(numeric_only=True)['G3'].sort_values()\n"
]

# Cell 25
nb['cells'][25]['source'] = [
    "# Calculate the absolute correlation of all features with the target variable G3\n",
    "correlations = mat_data.corr(numeric_only=True).abs()['G3'].sort_values(ascending=False)\n",
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

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched.")
