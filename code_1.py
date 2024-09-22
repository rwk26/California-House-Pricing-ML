from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import shap
# Load the California housing dataset
housing = fetch_california_housing()

# Convert it to a DataFrame for easier manipulation
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df['PRICE'] = housing.target

# Preview the data
print(housing_df.head())
#Checking for missing values:
print(housing_df.isnull().sum())
print(housing_df.describe())
#Building correlation heat map:
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(housing_df.corr(), annot=True, cmap='coolwarm')
plt.show()
#Standardazing our table in order to make our model prediction more precise, since the difference in features values is big
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(housing_df.drop('PRICE', axis=1))
#Building our model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, housing_df['PRICE'], test_size=0.2, random_state=42)
model = LinearRegression()

#Train the model using the training data
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model (optional: use R² score or mean squared error)
r2sc = r2_score(y_test, y_pred)
print(f'R2 score: {r2sc}')
#Predictions visualisation
y_pred = model.predict(X_test)

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()
#After analyzing visualization and R^2 parameter for the Linear regression model, I decided to try out the next model Random Forest Tree one, since I wasnt satisfied with Linear Regression R^2 parameter


#Define the Random Forest model
rf_model = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=2, random_state=42)

#Train the model
rf_model.fit(X_train, y_train)

#Make predictions
y_pred_rf = rf_model.predict(X_test)

#Evaluate the model
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest R2 Score: {r2_rf}')
#Since random forest model performed the best, lest take a look into a feature importances it chose
importances= rf_model.feature_importances_
features= housing.feature_names
feature_importance_df= pd.DataFrame({'Feature': features, 'Importance': importances})
print(feature_importance_df)
feature_importance_df=feature_importance_df.sort_values(by='Importance', ascending=False)
#Making a plot:
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Imporance in Random Forst Model')
plt.show()
# Cross-validate the Random Forest model
rf_cv_scores = cross_val_score(rf_model, X_scaled, housing_df['PRICE'], cv=5, scoring='r2')
print(f'Random Forest Cross-Validation R2 Scores: {rf_cv_scores}')
print(f'Average Cross-Validation R2 Score: {rf_cv_scores.mean()}')
#Set up the hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV to find the best combination of hyperparameters
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

#print(f'Best Parameters: {grid_search.best_params_}')

explainer= shap.TreeExplainer(rf_model)
X_test_sample=X_test[:500]
shap_values= explainer.shap_values(X_test_sample)
#Summary plot of features importance
shap.summary_plot(shap_values, X_test_sample, feature_names= housing.feature_names)