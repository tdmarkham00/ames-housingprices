# Packages required for modeling
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

train = pd.read_csv(
    "https://raw.githubusercontent.com/esnt/Data/main/Ames/ames_train.csv"
)
test = pd.read_csv(
    "https://raw.githubusercontent.com/esnt/Data/main/Ames/ames_test.csv"
)
pid = test["PID"]  # seperating the PID for the later export
X_test = test.iloc[:, 1:80]  # Removing PID from the feature list

# split training into X and y (test doesn't have y values)
X_train = train.iloc[:, 1:80]
ytrain = train["SalePrice"]

# combine X_train and X_test for standardization
X_all = pd.concat([X_train, X_test])

# Data cleaning
# NA values - fill w/ 0 or DNA
# Continuous features
cont = [
    "Lot Frontage",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Garage Yr Blt",
    "Garage Cars",
    "Garage Area",
]
# Categorical features
cat = [
    "Alley",
    "Mas Vnr Type",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Electrical",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Pool QC",
    "Fence",
    "Misc Feature",
]

X_all.loc[:, cont] = X_all.loc[:, cont].fillna(0)
X_all.loc[:, cat] = X_all.loc[:, cat].fillna("Does Not Apply")

# Creating dummy variables for the categorical
X_all = pd.get_dummies(X_all)
columns = (
    X_all.columns
)  # Pulling out column names to determine most important features later

# Standardize
scaler = StandardScaler()
Xall = scaler.fit_transform(X_all)

# split back into training/test
Xtrain = Xall[:1758, :]
Xtest = Xall[1758:, :]

l_cv = LassoCV(cv=10)
l_cv.fit(Xtrain, ytrain)

# Fit Lasso with best alpha
lasso = Lasso(alpha=l_cv.alpha_)
lasso.fit(Xtrain, ytrain)

# metrics for Lasso
# print(f"MSE: {round(mean_squared_error(ytrain, l_cv.predict(Xtrain)),4)}")
# print(f"RMSE: {round(np.sqrt(mean_squared_error(ytrain, l_cv.predict(Xtrain))),4)}")
# print(f"MAE: {round(mean_absolute_error(ytrain, l_cv.predict(Xtrain)),4)}")
# print(f"Y Standard Deviation (for comparison): {round(np.std(ytrain), 4)}")

# Predicted Y values
y_hat = lasso.predict(Xtest)

predictions = pd.DataFrame({"PID": pid, "Predictions": y_hat})

predictions.to_csv("Predictions.csv", index=False)
