import pandas as pd
import numpy as np
from sklearn import tree


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

target = train["Survived"].values

train["Age"] = train["Age"].fillna(train["Age"].median())
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

#Fitting the selected features in a decision tree
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

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

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

my_solution.to_csv("myPrediction.csv", index_label = ["PassengerId"])
