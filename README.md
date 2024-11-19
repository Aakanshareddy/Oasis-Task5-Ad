# Oasis-Task5-Ad

# Sales Prediction using Python - README

## Introduction
Sales prediction involves estimating future sales of a product based on features such as advertising spend across TV, radio, and newspapers. This helps businesses optimize their advertising strategies and forecast revenue effectively.

This project demonstrates the use of Python and Machine Learning to build a sales prediction model using linear regression.

---

## Project Workflow

### 1. **Data Preparation**
- **Dataset**: The dataset contains 200 records with the following features:
  - `TV`: Advertising expenditure on TV (in thousands of dollars).
  - `Radio`: Advertising expenditure on Radio.
  - `Newspaper`: Advertising expenditure on Newspapers.
  - `Sales`: Sales generated (target variable).
- **Data Cleaning**: Removed an unnecessary `Unnamed: 0` column.
- **Null Value Check**: No missing values were found.

### 2. **Exploratory Data Analysis**
- Data summary:
  - No null values.
  - Columns are numerical (`float64`).
- Key Statistics:
  - TV advertising spend ranges from **0.7 to 296.4**.
  - Radio spend ranges from **0 to 49.6**.
  - Newspaper spend ranges from **0.3 to 114**.
  - Sales range from **1.6 to 27**.
- **Visualization**: Pairplot (Seaborn) was used to analyze relationships between features and the target variable.

### 3. **Model Training**
- **Features** (`X`): `TV`, `Radio`, `Newspaper`.
- **Target** (`Y`): `Sales`.
- **Train-Test Split**: Data split into 80% training and 20% testing subsets.
- **Algorithm**: Linear Regression was used as the model.

### 4. **Model Evaluation**
- **Predictions**: The trained model was used to predict sales on the test dataset.
- **Evaluation Metrics**:
  - **Mean Squared Error (MSE)**: 3.174.
  - **R² Score**: 0.899, indicating a strong fit.

### 5. **Testing**
- Model tested with new inputs:
  - Example 1: `TV=200`, `Radio=50`, `Newspaper=10` → Predicted Sales: **21.41**.
  - Example 2: `TV=17.2`, `Radio=45.9`, `Newspaper=69.3` → Predicted Sales: **12.62**.

---

## Requirements
- **Libraries Used**:
  - `pandas`, `numpy`: Data handling and manipulation.
  - `matplotlib`, `seaborn`: Data visualization.
  - `sklearn`: Machine learning (Linear Regression, evaluation metrics, train-test split).

---

## Key Commands and Outputs
1. **Loading Data**:
   ```python
   df = pd.read_csv("/content/Advertising.csv")
   df = df.drop(['Unnamed: 0'], axis=1)
   ```
2. **Checking for Nulls**:
   ```python
   df.isnull().sum()
   ```
   Output: No null values.
3. **Model Training**:
   ```python
   model = LinearRegression()
   model.fit(X_train, Y_train)
   ```
4. **Evaluation**:
   ```python
   mean_squared_error(Y_test, y_pred)  # Output: 3.174
   r2_score(Y_test, y_pred)           # Output: 0.899
   ```
5. **Prediction**:
   ```python
   new_data = pd.DataFrame({'TV': [200], 'Radio': [50], 'Newspaper': [10]})
   model.predict(new_data)  # Output: [21.41]
   ```

---

## Results and Insights
- TV and Radio advertising have a stronger impact on sales compared to Newspaper advertising.
- The linear regression model performs well, with a high R² score (0.899), indicating it explains ~90% of the variability in sales.

---

## Future Improvements
- **Feature Engineering**: Explore interaction terms between features.
- **Advanced Models**: Implement other algorithms (e.g., Decision Trees, Random Forests).
- **Hyperparameter Tuning**: Optimize the linear regression model.

--- 

## Usage
Run the script in a Python environment (e.g., Jupyter Notebook, Google Colab) with the required libraries installed. Modify the input data for predictions as needed.

--- 

## Conclusion
This project demonstrates the use of linear regression for predicting sales based on advertising expenditure. It provides a framework for building and evaluating regression models for business forecasting.
