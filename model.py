import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("house_data.csv")
print(df.head())

# Selecting numeric columns for feature set
numeric_columns = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
                   "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
                   "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]
X = df[numeric_columns]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling: normalises data acc to a common scale
sc = StandardScaler()
#calculate mean and standard deviation if the training dataset
X_train = sc.fit_transform(X_train)
#scaling or tranforming the test data in the same way as the trainig data acc to the the mean and sd of the training data

X_test = sc.transform(X_test)

# Instantiate the model
regressor = RandomForestRegressor()

# Fit the model
regressor.fit(X_train, y_train)

# Make pickle file of the model
pickle.dump(regressor, open("model.pkl", "wb"))
