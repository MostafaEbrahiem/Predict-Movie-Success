import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
def polynomial_reg(Movies_training):

    X1 = Movies_training.iloc[:, 0:34]  # X1 for multi linear reg Features
    Y1 = Movies_training['IMDb']  # Label FOR MULTI LINEAR

    # Split the data to training and testing sets
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.20, shuffle=True)

    poly_reg_start_time = time.time()
    poly_features = PolynomialFeatures(degree=2)
    X1_train_poly = poly_features.fit_transform(X1_train)
    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X1_train_poly, Y1_train)
    # predicting on training data-set
    y_train_predicted = poly_model.predict(X1_train_poly)
    # predicting on test data-set
    prediction3 = poly_model.predict(poly_features.fit_transform(X1_test))
    poly_reg_end_time = time.time()
    print('polynomial rag time = ', poly_reg_end_time - poly_reg_start_time)

    print('Co-efficient of Polynomial regression', poly_model.coef_)
    print('Intercept of  Polynomial regression model', poly_model.intercept_)
    print('Mean Square Error of  Polynomial regression model', metrics.mean_squared_error(Y1_test, prediction3))
    # print('Accuracy of Polynomial regression',poly_model.score(X1_test,Y1_test))
    true_rate_value = np.asarray(Y1_test)[0]
    predicted_rate_value = prediction3[0]
    print('True value for the first movie in the test is : ' + str(true_rate_value))
    print('Predicted value for the first movie in the test set is : ' + str(predicted_rate_value))
