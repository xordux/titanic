import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


train_url = "trainingData/train.csv"
train = pd.read_csv(train_url)

test_url = "trainingData/test.csv"
test = pd.read_csv(test_url)

train["Sex"][train["Sex"] != "male"] = 1 # all females and empty entries will be 1
train["Sex"][train["Sex"] == "male"] = 0

#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

train["Age"] = train["Age"].fillna(train["Age"].median())

# Adding a feature to improve prediction:
# train["family_size"] = train["SibSp"].values + train["Parch"].values + 1

feature_names = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]

target = train["Survived"].values
features_one = train[feature_names].values

#Fitting the selected features in a decision tree
max_depth = 10
min_samples_split = 2
my_forest = RandomForestClassifier(max_depth = max_depth, min_samples_split = min_samples_split, n_estimators = 100, random_state = 1)
my_forest = my_forest.fit(features_one, target)

# -------------------------------------------------------------------------------
# ------------------------------- Test cases ------------------------------------
# -------------------------------------------------------------------------------
# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

test["Sex"][test["Sex"] != "male"] = 1 # all females and empty entries will be 1
test["Sex"][test["Sex"] == "male"] = 0

#Impute the Embarked variable
test["Embarked"] = test["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


test["Age"] = test["Age"].fillna(test["Age"].median())

test["family_size"] = test["SibSp"].values + test["Parch"].values + 1
# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[feature_names].values

# Make your prediction using the test set
my_prediction = my_forest.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

my_solution.to_csv("myPrediction.csv", index_label = ["PassengerId"])
