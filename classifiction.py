from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pickle import dump,load
from sklearn.metrics import f1_score

def classifiction (c_Movies_training):
    X1 = c_Movies_training.iloc[:, 1:-1]  # X1 for multi linear reg Features
    Y1 = c_Movies_training['rate']  # Label FOR MULTI LINEAR

    # Split the data to training and testing sets
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.99, shuffle=True)
    ada_bost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME.R",
                             n_estimators=200)


    ada_bost = load(open('ada_bost.pkl', 'rb'))

    #ada_bost.fit(X1_train, Y1_train)
    #dump(ada_bost, open('ada_bost.pkl', 'wb'))

    y_prediction = ada_bost.predict(X1_test)
    accuracy = np.mean(y_prediction == Y1_test) * 100

    print("The achieved accuracy using Adaboost is " + str(accuracy))
    #print(str(f1_score(Y1_test,y_prediction)))

    dtree_model = load(open('dtree_model.pkl', 'rb'))
    #dtree_model = DecisionTreeClassifier(max_depth=10).fit(X1_train, Y1_train)
    #dump(dtree_model, open('dtree_model.pkl', 'wb'))
    dtree_predictions = dtree_model.predict(X1_test)

    #accuracy1=dtree_model.score(X1_test,Y1_test)
    accuracy1=np.mean(dtree_predictions == Y1_test) * 100
    print("The achieved accuracy using DecisionTree is " + str(accuracy1))
    # creating a confusion matrix
    cm = confusion_matrix(Y1_test, dtree_predictions)
    #///////////////////////////////////////////////////////////////////////////////////////////////////
    #svm_model_linear = SVC(kernel='linear', C=1).fit(X1_train, Y1_train)
    #svm_predictions = svm_model_linear.predict(X1_test)
    C=0.3
    #poly_svc = SVC(kernel='poly', degree=5, C=C).fit(X1_train, Y1_train)
    #poly_predicyion=poly_svc.predict(X1_test)

    rbf_svc = load(open('rbf_svc.pkl', 'rb'))
    #rbf_svc = SVC(kernel='rbf', gamma=0.8, C=C).fit(X1_train, Y1_train)
    #dump(rbf_svc, open('rbf_svc.pkl', 'wb'))
    rbf_svc_predection=rbf_svc.predict(X1_test)


    # model accuracy for X_test

    #accuracy2=poly_svc.score(X1_test,Y1_test)
    #accuracy2 = svm_model_linear.score(X1_test, Y1_test)
    accuracy2=rbf_svc.score(X1_test,Y1_test)*100
    print("The achieved accuracy using SVM is " + str(accuracy2))
    # creating a confusion matrix
    #cm = confusion_matrix(Y1_test, svm_predictions)
    #cm=confusion_matrix(Y1_test,poly_predicyion)
    cm=confusion_matrix(Y1_test,rbf_svc_predection)
    #///////////////////////////////////////////////////////////////

    knn = load(open('knn.pkl', 'rb'))
    #knn = KNeighborsClassifier(n_neighbors=20).fit(X1_train, Y1_train)
    #dump(knn, open('knn.pkl', 'wb'))
    # accuracy on X_test
    accuracy3 = knn.score(X1_test, Y1_test)*100
    print("The achieved accuracy using KNeighbors is " + str(accuracy3))

    # creating a confusion matrix
    knn_predictions = knn.predict(X1_test)
    cm = confusion_matrix(Y1_test, knn_predictions)
