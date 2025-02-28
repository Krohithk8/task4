import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the Dataset
data = pd.read_csv('car_price_prediction.csv')  # Replace with the actual path to the dataset

# Step 2: Data Exploration
print("Dataset Overview:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 3: Check for Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 4: Handle Missing Values (if any)
# You can either drop rows or fill them with the mean/median/mode
data = data.dropna()  # Drop rows with missing values (simple approach)

# Step 5: Feature Engineering - Convert Categorical Columns to Numerical
# Use Label Encoding to convert categorical variables (e.g., car brands, fuel types)
label_encoder = LabelEncoder()
categorical_columns = ['make', 'model', 'fuel_type']  # Replace with your dataset's categorical columns

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Step 6: Feature Selection
# Select the features (X) and target variable (y)
X = data.drop('price', axis=1)  # Assuming 'price' is the target column
y = data['price']

# Step 7: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Training - Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Model Prediction
y_pred = model.predict(X_test)

# Step 10: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Step 11: Print the Evaluation Metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared (R2): {r2}")

# Step 12: Visualize Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted Car Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Step 13: Train a Linear Regression Model (for comparison)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict with Linear Regression
y_pred_linear = linear_model.predict(X_test)

# Evaluate Linear Regression Model
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Print Linear Regression Metrics
print("\nLinear Regression Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae_linear}")
print(f"Mean Squared Error (MSE): {mse_linear}")
print(f"Root Mean Squared Error (RMSE): {rmse_linear}")
print(f"R-Squared (R2): {r2_linear}")

# Visualize Actual vs Predicted Prices for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear)
plt.title('Actual vs Predicted Car Prices (Linear Regression)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Step 14: Save the Model (Optional)
import joblib
joblib.dump(model, 'car_price_model_rf.pkl')  # Save the Random Forest model
joblib.dump(linear_model, 'car_price_model_lr.pkl')  # Save the Linear Regression model
