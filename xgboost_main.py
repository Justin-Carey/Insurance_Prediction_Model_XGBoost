import pandas as pd
import numpy as np
import xgboost as xgb
import winsound
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

# Using the features in the dataset, predict if an insurace claim will occur or not.
def xgboost_classifier(X_train, X_test, Y_train):

    # For training, if the target variable is above 0, convert it to class label "1".
    Y_train = np.where(Y_train > 0, 1, 0)

    model = xgb.XGBClassifier(max_depth=20, learning_rate=.15, n_estimators=400)     # 30%

    # Classes are imbalanced (more zeroes than ones), add weight to the ones to offset.
    sw = ((Y_train * 15) - Y_train) + np.ones(len(Y_train))     # 46.7%

    model.fit(X_train, Y_train, sample_weight=sw)

    preds = model.predict(X_test)

    return preds

# Predict how much a claim is going to be. 
def xgboost_regressor(X_train, X_test, Y_train):

    model = xgb.XGBRegressor(max_depth=20, learning_rate=0.001)     # 98.56

    model.fit(X_train, Y_train)

    preds = model.predict(X_test)

    return preds

# Filter the results by multiplying the results of the classifier with the predictor.
def run_class_regress(df):

    X = df.drop(['ClaimAmount', 'rowIndex'], axis=1)
    Y = df.ClaimAmount

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

    regress_preds = xgboost_regressor(X_train, X_test, Y_train)
    class_preds = xgboost_classifier(X_train, X_test, Y_train)

    result = regress_preds * class_preds

    print("Test set Mae:", mean_absolute_error(Y_test, result))

    result = np.where(result > 0, 1, 0)
    Y_test = np.where(Y_test > 0, 1, 0)

    print("Test set F1:", f1_score(Y_test, result))
    print("Test set Confusion:\n ", confusion_matrix(Y_test, result))


def main():

    df = pd.read_csv("trainingset.csv")

    start = timer()
    run_class_regress(df)
    end = timer()
    print("Took", end - start, "seconds")

    winsound.Beep(250, 1000)


if __name__ == "__main__":
    main()

