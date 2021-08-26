from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import  time
import matplotlib.pyplot as plt
from pickle import dump,load
from sklearn.metrics import r2_score, f1_score

def lin_reg(Movies_training):
    X = Movies_training['Rotten Tomatoes']  # X for single linear reg
    Y = Movies_training['IMDb']  # Label FOR SIGLE LINEAR
    # Split the data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1, shuffle=True)

    linear_reg_start_time=time.time()
    cls = linear_model.LinearRegression()
    X_train = np.expand_dims(X_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    #train
    #cls.fit(X_train, y_train)
    #dump(cls, open('linear_reg.pkl', 'wb'))
    cls = load(open('linear_reg.pkl', 'rb'))
    prediction = cls.predict(X_test)
    #prediction = cls.predict(X_test)
    linear_reg_end_time=time.time()
    plt.scatter(X_train, y_train)
    plt.xlabel('Rotten Tomatoes', fontsize=20)
    plt.ylabel('IMDb', fontsize=20)
    plt.plot(X_test, prediction, color='red', linewidth=3)
    plt.show()
    print('linear rag time = ', linear_reg_end_time-linear_reg_start_time)
    # //////////print('Co-efficient of SINGLE linear regression',cls.coef,MSE,ACCURACY_)////////////

    #print('coef of Single regression model', cls.coef_)
    #print('Intercept of Single regression model', cls.intercept_)
    print('Mean Square Error in Single linear regression', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print('Accuracy of Single regression', cls.score(y_test, prediction))
    print('r2 score of Single regression', metrics.r2_score(y_test, prediction))
    true_rate_value = np.asarray(y_test)[0]
    predicted_rate_value = prediction[0]
    print('True value for the first movie in the test is : ' + str(true_rate_value))
    print('Predicted value for the first movie in the test set is : ' + str(predicted_rate_value))


