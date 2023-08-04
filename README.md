# House-Price-Prediction
House Price Prediction Model using Random Forest Regressor
This repository contains a machine learning model for predicting house prices based on various features using the Random Forest Regressor algorithm. The model is implemented in Python using the Pandas library for data handling, Scikit-learn for machine learning tools, and Pickle for model serialization.

Dataset
The dataset used for training and testing the model is stored in a CSV file named "house_data.csv". It contains information about various houses, such as the number of bedrooms, bathrooms, square footage, location attributes, and other relevant features.

Model Creation
The model is built using the Random Forest Regressor, a powerful ensemble learning method for regression tasks. The following steps outline the model creation process:

Data Preprocessing: The numeric columns from the dataset are selected as the feature set (X) for the model, and the target variable is the "price" column (y).

Data Splitting: The dataset is split into training and testing sets, with 70% of the data used for training and 30% for testing.

Feature Scaling: To ensure that all features have the same scale and do not dominate the others during model training, the data is standardized using the StandardScaler from Scikit-learn.

Model Training: The Random Forest Regressor is instantiated, and it is trained on the standardized training data (X_train and y_train).

Model Serialization: Once the model is trained, it is serialized using Pickle and saved as "model.pkl". This allows us to easily load the trained model and make predictions on new data without retraining.
