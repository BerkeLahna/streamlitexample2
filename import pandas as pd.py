import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')
df = pd.read_csv('bank-additional.csv', delimiter=';')
df.dropna(inplace=True)
numeric_cols = ['duration', 'campaign', 'pdays', 'previous', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_mask = (df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)

df = df[(df[numeric_cols] > lower_bound) & (df[numeric_cols] < upper_bound)]

cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline

# Split the dataset into features (X) and target variable (y)
X = df.drop('y', axis=1)
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the scalers to evaluate
scalers = [
    ('StandardScaler', StandardScaler()),
    ('MinMaxScaler', MinMaxScaler()),
    ('RobustScaler', RobustScaler()),
    ('Normalizer', Normalizer())
]

# Create the logistic regression classifier
model = LogisticRegression()

# Define the parameter grid for grid search
param_grid = {
    'model__C': [0.1, 1.0, 10.0],  # example parameter values to search over
    'model__penalty': ['l1', 'l2']
}

best_accuracy = 0.0
best_scaler = None

# Iterate over the scalers and perform grid search with cross-validation
for scaler_name, scaler in scalers:
    # Create a pipeline with scaler and model
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Calculate the accuracy using the scaler
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy with', scaler_name, ':', accuracy)

    # Check if the current scaler gives a better accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_scaler = scaler_name

print('Best scaler is', best_scaler)
print('Best accuracy score is', best_accuracy)

import streamlit as st

# ...

# Create a function to preprocess the input data
def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data)
    for col in cols:
        input_df[col] = le.transform(input_df[col])
    return input_df

# Create a function to make predictions
def make_predictions(input_data):
    scaled_input = best_model.named_steps['scaler'].transform(input_data)
    predictions = best_model.predict(scaled_input)
    return predictions[0]

# Create the Streamlit app
def main():
    st.title("Classification Model App")
    st.write("Enter the parameters below to make predictions.")

    # Create input fields
    input_data = {}
    for col in X.columns:
        if col != 'y':
            if col in numeric_cols:
                input_data[col] = st.number_input(col, value=0)
            else:
                input_data[col] = st.text_input(col)

    # Preprocess the input data
    input_df = preprocess_input([input_data])

    # Make predictions when the user clicks the "Predict" button
    if st.button("Predict"):
        prediction = make_predictions(input_df)
        st.write("Predicted class:", prediction)

# Run the Streamlit app
if __name__ == '__main__':
    main()
