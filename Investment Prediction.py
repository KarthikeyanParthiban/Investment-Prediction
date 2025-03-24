# %% [markdown]
# ### Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# %% [markdown]
# ### Data Generation Functions and DataFrame Creation:

# %%

np.random.seed(42)


num_rows = 155411


def generate_initial_amount(num_rows):
    amounts = []
    for _ in range(num_rows):
        rand_val = np.random.rand()
        if rand_val < 0.55:
            amount = np.random.choice([5, 10, 15, 20, 25, 30, 35, 40]) + np.random.randint(-3, 6)
        elif rand_val < 0.85:
            amount = np.random.choice([57, 26, 105, 24]) + np.random.randint(-7, 12)
        else:
            amount = np.random.choice([150, 380, 2400, 1750]) + np.random.randint(-50, 100)
        amounts.append(max(1, amount))
    return [amt * 1000 for amt in amounts]


df = pd.DataFrame({
    "Initial_Selected_Amount": generate_initial_amount(num_rows),
    "tenure_months": np.random.choice([12, 24, 36, 48, 60], num_rows, p=[0.33, 0.18,0.15,0.26,0.08]),
    "gender": np.random.choice(["Female", "Male"], num_rows, p=[0.62, 0.38]),
    "Customer Type": np.random.choice(["Existing", "New"], num_rows, p=[0.72, 0.28])
})

df["age"] = np.random.randint(22, 80, num_rows)

noise = df["Initial_Selected_Amount"] * np.random.normal(0, 0.2, num_rows)
market_volatility = np.random.normal(0.05, 0.15, num_rows)
customer_effect = np.where(df["Customer Type"] == "Existing", 0.03 * df["tenure_months"], -0.015 * df["tenure_months"])
gender_effect = np.where(df["gender"] == "Female", 0.012 * df["tenure_months"], -0.007 * df["tenure_months"])

df["Invested Amount"] = df["Initial_Selected_Amount"] + noise + (df["tenure_months"] * np.random.randint(200, 300))
df["Invested Amount"] *= (1 + market_volatility + customer_effect / 100 + gender_effect / 100)

dynamic_fees = np.clip(np.random.normal(0.012, 0.006, num_rows), 0.005, 0.02)
df["Invested Amount"] *= (1 - dynamic_fees)

df["Invested Amount"] = (df["Invested Amount"] // 10000) * 10000
df["Invested Amount"] = df["Invested Amount"].apply(lambda x: max(x, 10000))

df.head()


# %% [markdown]
# ### Data Exploration

# %%
print("Dataframe Info:")
print(df.info())

# %%
print("\nDescriptive Statistics:")
print(df.describe())

# %%
# Convert categorical to numeric

df["is_senior_citizen"] = df['age'].apply(lambda x: 1 if x>=60 else 0)
df["is_women"] = df['gender'].apply(lambda x: 1 if x=='Female' else 0)
df["is_ETB"] = df['Customer Type'].apply(lambda x: 1 if x=='Existing' else 0)


df.drop(columns=['age','gender','Customer Type'],inplace=True)

# %%
df.hist(figsize=(12, 10), bins=20)  
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()

# %%
# Correlation Matrix - 

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# %%
# Pairplot to visualize relationships between features 

sns.pairplot(df[["Initial_Selected_Amount", "tenure_months", "Invested Amount"]])
plt.suptitle("Pairplot of Key Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
# categorical features

categorical_cols = ['is_senior_citizen', 'is_women', 'is_ETB']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Data Splitting

# %%

x = df.columns.drop(['Invested Amount'])
y = df['Invested Amount']


X = df[x]
y = df['Invested Amount']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Model Training and Evaluation

# %%
# Model Training and Evaluation (XGBoost)


xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42,  
                     max_depth=5,  
                     subsample=0.8,  
                     colsample_bytree=0.8) 
xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test)

y_pred_xgb = np.round(y_pred_xgb / 10000) * 10000

# Accuracy Metrics
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)  
rmse_xgb = np.sqrt(mse_xgb)  

print("XGBoost Model Performance:")
print(f"R²: {r2_xgb:.2f}")
print(f"MAE: {mae_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")

# %% [markdown]
# ### Visualization and Feature Importance

# %%
# actual vs predicted values

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) #Ideal line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (XGBoost)")
plt.show()

# %%
# Feature Importance
feature_importance = xgb.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

# %% [markdown]
# ### Example Prediction

# %%
# Example Prediction

example_data = pd.DataFrame({
    "Initial_Selected_Amount": [45000],
    "tenure_months": [60],
    "is_senior_citizen": [0],
    "is_women": [1],
    "is_ETB": [0]
})


example_data = example_data[X.columns]

example_prediction = xgb.predict(example_data)
example_prediction = np.round(example_prediction / 10000) * 10000
print(f"Predicted Total Amount: {example_prediction[0]:.0f}")


