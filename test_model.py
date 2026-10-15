import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

mat_data = pd.read_csv('student-mat.csv')
mat_data = pd.get_dummies(mat_data, drop_first=True, dtype=int)

correlations = mat_data.corr(numeric_only=True).abs()['G3'].sort_values(ascending=False)
top_features = correlations[:10].index.tolist()

required_features = ['G1', 'G2', 'studytime', 'absences', 'G3']
for feature in required_features:
    if feature not in top_features and feature in mat_data.columns:
        top_features.append(feature)

mat_data = mat_data[top_features]

X = mat_data.drop('G3', axis='columns')
y = mat_data['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f'R2: {r2}')
