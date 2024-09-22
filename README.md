
# California Housing Price Prediction using Machine Learning

## Project Overview
This project aims to predict housing prices in California using data from the `California Housing Dataset`. The project explores both **Linear Regression** and **Random Forest Regression** models to understand which model performs better in predicting house prices. Furthermore, **SHAP** values are used to explain the model's predictions and understand the importance of each feature.

## Project Motivation
The goal of this project is to build a machine learning model that accurately predicts housing prices based on key features such as median income, house age, and geographical location. Understanding which features influence house prices can provide valuable insights into the California housing market.

## Dataset
The dataset used in this project is the **California Housing Dataset**, which contains information collected from the 1990 U.S. Census. The target variable is `PRICE`, which represents the median house value in a particular region.

### Features:
- **MedInc**: Median income in the region
- **HouseAge**: Average age of houses in the region
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Population of the region
- **AveOccup**: Average number of occupants per household
- **Latitude**: Latitude of the region
- **Longitude**: Longitude of the region

## Project Workflow
1. **Data Preprocessing**: 
   - The data is loaded and transformed into a pandas DataFrame.
   - Checked for missing values and conducted exploratory data analysis (EDA), including correlation heatmaps.
   - Standardized the features to ensure consistent scaling across different features.
   
2. **Model Building**:
   - Two models were built and compared:
     1. **Linear Regression**: A simple linear model that assumes a linear relationship between the features and the target.
     2. **Random Forest Regression**: An ensemble learning method that builds multiple decision trees to improve prediction accuracy.
     
3. **Model Evaluation**:
   - Evaluated the models using **Mean Squared Error (MSE)** and **R² score** on the test set.
   - **Linear Regression** showed a lower R² score (~0.55), while the **Random Forest Regressor** performed better with an average R² score of **0.66** using cross-validation.
   
4. **Feature Importance and Explainability**:
   - Feature importance was analyzed using the Random Forest model.
   - **SHAP (SHapley Additive exPlanations)** values were calculated to explain the model's predictions and show how each feature impacted individual predictions.

## Results

### Model Performance:
- **Linear Regression**:
  - R² Score: ~0.55
  
- **Random Forest Regression**:
  - R² Score: ~0.80 (on the same test sample set from regression model)
  - Cross-Validation R² Scores: [0.514, 0.703, 0.742, 0.636, 0.682]
  - Average Cross-Validation R² Score: **0.66**
  - Hyperparameter tuning using **GridSearchCV** found the best parameters:
    - `max_depth`: 30
    - `min_samples_split`: 2
    - `n_estimators`: 300
  
### Feature Importance:
- The **median income (MedInc)** was the most important feature, contributing significantly to the model's predictions.
- Geographical features such as **Latitude** and **Longitude** also played an important role in determining house prices.

### SHAP Analysis:
- **SHAP values** were used to explain the model’s predictions.
- Key insights:
  - Higher **MedInc** values push house prices higher, while lower **MedInc** values push them lower.
  - Geographical factors such as **Latitude** and **Longitude** had notable impacts, indicating that location plays a critical role in house pricing.
  - Features like **AveOccup** (average number of occupants per household) also influenced house prices to some degree.

### Conclusion:
The **Random Forest Regressor** provided better performance compared to the **Linear Regression** model. The model also highlighted that **median income** is the strongest predictor of housing prices, followed by geographical location. Using **SHAP** values helped explain the model's predictions and made the feature importance more interpretable.

## Files in This Repository
- code_1.py: Contains the entire workflow, including data preprocessing, model training, evaluation, and SHAP analysis.
- Figures: All visualizations used in this project.
- `README.md`: Project documentation and summary.


## Requirements
To run this project locally, you need the following Python libraries:
- pandas
- scikit-learn
- matplotlib
- seaborn
- shap
