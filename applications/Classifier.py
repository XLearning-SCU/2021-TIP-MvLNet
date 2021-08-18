import sklearn.metrics as SM
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


def GBC(train_data, train_label, test_data, test_label, verbose=True, decimals=4):
    clf = GradientBoostingClassifier()
    clf.fit(train_data, train_label)
    pre_label = clf.predict(test_data)

    ret = {}
    ret["accuracy"] = np.round(SM.accuracy_score(test_label, pre_label), decimals)
    ret["precision"] = np.round(
        SM.precision_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["recall"] = np.round(
        SM.recall_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["f_measure"] = np.round(
        SM.f1_score(test_label, pre_label, average="weighted"), decimals
    )
    if verbose:
        print("Method: GBC")
        print_scores(ret["accuracy"], ret["precision"], ret["recall"], ret["f_measure"])
    return ret


def DecisionTree(
    train_data, train_label, test_data, test_label, verbose=True, decimals=4
):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_label)
    pre_label = clf.predict(test_data)

    ret = {}
    ret["accuracy"] = np.round(SM.accuracy_score(test_label, pre_label), decimals)
    ret["precision"] = np.round(
        SM.precision_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["recall"] = np.round(
        SM.recall_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["f_measure"] = np.round(
        SM.f1_score(test_label, pre_label, average="weighted"), decimals
    )
    if verbose:
        print("Method: DT")
        print_scores(ret["accuracy"], ret["precision"], ret["recall"], ret["f_measure"])
    return ret


def knn_score(
    train_data, train_label, test_data, test_label, knn_k=1, verbose=True, decimals=4
):
    knn = neighbors.KNeighborsClassifier(n_neighbors=knn_k)
    knn.fit(train_data, train_label)
    pre_label = knn.predict(test_data)

    ret = {}
    ret["accuracy"] = np.round(SM.accuracy_score(test_label, pre_label), decimals)
    ret["precision"] = np.round(
        SM.precision_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["recall"] = np.round(
        SM.recall_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["f_measure"] = np.round(
        SM.f1_score(test_label, pre_label, average="weighted"), decimals
    )
    if verbose:
        print("Method: KNN")
        print_scores(ret["accuracy"], ret["precision"], ret["recall"], ret["f_measure"])
    return ret


def SVM(train_data, train_label, test_data, test_label, verbose=True, decimals=4):
    clf = svm.LinearSVC()
    clf.fit(train_data, train_label)
    pre_label = clf.predict(test_data)

    ret = {}
    ret["accuracy"] = np.round(SM.accuracy_score(test_label, pre_label), decimals)
    ret["precision"] = np.round(
        SM.precision_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["recall"] = np.round(
        SM.recall_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["f_measure"] = np.round(
        SM.f1_score(test_label, pre_label, average="weighted"), decimals
    )
    if verbose:
        print("Method: SVM")
        print_scores(ret["accuracy"], ret["precision"], ret["recall"], ret["f_measure"])
    return ret


def MLP(train_data, train_label, test_data, test_label, verbose=True, decimals=4):
    mlp = MLPClassifier()
    mlp.fit(train_data, train_label)
    pre_label = mlp.predict(test_data)

    ret = {}
    ret["accuracy"] = np.round(SM.accuracy_score(test_label, pre_label), decimals)
    ret["precision"] = np.round(
        SM.precision_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["recall"] = np.round(
        SM.recall_score(test_label, pre_label, average="weighted"), decimals
    )
    ret["f_measure"] = np.round(
        SM.f1_score(test_label, pre_label, average="weighted"), decimals
    )
    if verbose:
        print("Method: MLP")
        print_scores(ret["accuracy"], ret["precision"], ret["recall"], ret["f_measure"])
    return ret


def print_scores(accuracy, precision, recall, f_measure):
    print(
        "ACC",
        accuracy,
        "Precision",
        precision,
        "Recall",
        recall,
        "F-measure",
        f_measure,
    )
