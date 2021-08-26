from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import  time
from sklearn import svm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def SVC_reg(Movies_training):
    X = Movies_training.iloc[:,2:3]
    Y= Movies_training['IMDb']
    X=np.array(X).reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)
   # plt.scatter(X_train ,Y_train,s=5,color="blue")

    #plt.show()

    svr=SVR().fit(X_train,Y_train)
    predection=svr.predict(X_test)
    plt.scatter(X_train, Y_train, s=5, color="blue",label="original")
    plt.plot(X_test,predection, lw=2, color="red", label="SVC")
    plt.legend()
    plt.show()
    print('SVM MSE using single data = ',mean_squared_error(Y_test,predection))

