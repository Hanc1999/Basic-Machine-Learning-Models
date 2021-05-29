import numpy as np
import matplotlib.pyplot as plt
import graphviz
import pandas as pd

from sklearn import tree
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error


feature = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'ShelveLoc', 'Age',
           'Education', 'Urban', 'US']


def load_data(filename):
    temp = pd.read_csv(filename)
    # Transform string values to numeric
    temp['ShelveLoc'] = temp['ShelveLoc'].map({'Bad': 0, 'Medium': 1, 'Good': 2})
    temp['Urban'] = (temp['Urban'] == 'Yes').astype("int")
    temp['US'] = (temp['US'] == 'Yes').astype("int")
    # Obtain statistics for the dataset and plot
    # statistics(temp)
    x_train = temp.iloc[0:300, 1:]
    y_train = temp.iloc[0:300, 0]
    x_test = temp.iloc[300:, 1:]
    y_test = temp.iloc[300:, 0]
    return x_train, y_train, x_test, y_test


# def statistics(temp):
    ''' Plot the histograms for target variables and features '''
    # temp.hist(column='Sales')
    # temp.hist(column=['CompPrice', 'Income', 'Advertising', 'Population'])
    # temp.hist(column=['Price', 'ShelveLoc', 'Age', 'Education'])
    # temp.hist(column=['Urban', 'US'], bins=3)
    # plt.show()

    ''' Calculate the statistics(mean, median, variance) for target variables and features'''
    # print(temp.mean(axis=0))
    # print(temp.median(axis=0))
    # print(temp.var(axis=0))


def decision_tree(x_train, y_train, x_test, y_test, dep=5, lsn=1):
    clf = tree.DecisionTreeRegressor(max_depth=dep, min_samples_leaf=lsn)
    clf = clf.fit(x_train, y_train)
    print(f"\nDecision tree ~ depth={dep} least node sample={lsn}")
    y_pred = clf.predict(x_train)
    print(f"Train error:{mean_squared_error(y_pred, y_train):.4f}")
    y_pred = clf.predict(x_test)
    print(f"Test error:{mean_squared_error(y_pred, y_test):.4f}")

    fI = clf.feature_importances_
    for i in range(len(fI)):
        print(f"{feature[i]}:{fI[i]:.2f}")

    tree.export_graphviz(clf, out_file='a.dot', feature_names=feature, filled=True)
    # dot -Tpng a.dot -o tree.png
    # figure = tree.plot_tree(clf)


def bagging(x_train, y_train, x_test, y_test, dep=5, numTree=100):
    base = tree.DecisionTreeRegressor(max_depth=dep, min_samples_leaf=5)
    clf = BaggingRegressor(base, n_estimators=numTree, oob_score=True)
    clf = clf.fit(x_train, y_train)
    print(f"\nBagging ~ depth={dep} number of trees={numTree}")
    y_pred = clf.predict(x_train)
    print(f"Train error:{mean_squared_error(y_pred, y_train):.4f}")
    y_pred = clf.predict(x_test)
    print(f"Test error:{mean_squared_error(y_pred, y_test):.4f}")
    bias = abs(np.mean((y_test - np.mean(y_pred))))
    print(f"Bias:{bias:.4f}")
    variance = y_pred.var(axis=0)
    print(f"Variance:{variance:.4f}")

    # fI = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    # for i in range(len(fI)):
    #     print(f"{feature[i]}:{fI[i]:.2f}")

    score = clf.oob_score_
    print(f"OOB Score:{score:.4f}")


def randomForest(x_train, y_train, x_test, y_test, numTree=100, mtry=10):
    clf = RandomForestRegressor(n_estimators=numTree, max_features=mtry, oob_score=True)
    clf = clf.fit(x_train, y_train)
    print(f"\nRandom forest ~ number of trees={numTree} mtry={mtry}")
    y_pred = clf.predict(x_train)
    print(f"Train error:{mean_squared_error(y_pred, y_train):.4f}")
    y_pred = clf.predict(x_test)
    print(f"Test error:{mean_squared_error(y_pred, y_test):.4f}")
    bias = abs(np.mean((y_test - np.mean(y_pred))))
    print(f"Bias:{bias:.4f}")
    variance = y_pred.var(axis=0)
    print(f"Variance:{variance:.4f}")

    # fI = clf.feature_importances_
    # for i in range(len(fI)):
    #     print(f"{feature[i]}:{fI[i]:.2f}")

    score = clf.oob_score_
    print(f"OOB Score:{score:.4f}")


def adaBoost(x_train, y_train, x_test, y_test, round=100):
    clf = AdaBoostRegressor(n_estimators=round)
    clf = clf.fit(x_train, y_train)
    print(f"\nAdaBoost ~ round={round}")
    y_pred = clf.predict(x_train)
    print(f"Train error:{mean_squared_error(y_pred, y_train):.4f}")
    y_pred = clf.predict(x_test)
    print(f"Test error:{mean_squared_error(y_pred, y_test):.4f}")
    bias = abs(np.mean((y_test - np.mean(y_pred))))
    print(f"Bias:{bias:.4f}")
    variance = y_pred.var(axis=0)
    print(f"Variance:{variance:.4f}")

    # fI = clf.feature_importances_
    # for i in range(len(fI)):
    #     print(f"{feature[i]}:{fI[i]:.2f}")


x_train, y_train, x_test, y_test = load_data("Carseats.csv")

# Decision tree
depth = [1, 3, 5, 7, 9]
lsn = [1, 5, 9, 13, 17]
for i in depth:
    decision_tree(x_train, y_train, x_test, y_test, i, 1)
for i in lsn:
    decision_tree(x_train, y_train, x_test, y_test, 5, i)

# Bagging
depth = [1, 3, 5, 7, 9]
numTree = [50, 75, 100, 125, 150]
for i in depth:
    bagging(x_train, y_train, x_test, y_test, i, 100)
for i in numTree:
    bagging(x_train, y_train, x_test, y_test, 5, i)

# Random Forest
numTree = [50, 75, 100, 125, 150]
mtry = [2, 4, 6, 8, 10]
for i in numTree:
    randomForest(x_train, y_train, x_test, y_test, i, 10)
for i in mtry:
    randomForest(x_train, y_train, x_test, y_test, 100, i)

# AdaBoost
round = [50, 100, 150, 200, 250]
for i in round:
    adaBoost(x_train, y_train, x_test, y_test, i)
