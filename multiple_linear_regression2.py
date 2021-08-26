import seaborn as sns
import matplotlib.pyplot as plt
from linear_regression import *
import  time

def Mult_reg(Movies_training):
    X1 = Movies_training.iloc[:, 1:35]  # X1 for multi linear reg Features
    Y1=Movies_training['IMDb'] #Label FOR MULTI LINEAR

    # Split the data to training and testing sets
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.20,shuffle=True)

    #Get the correlation between the features
    corr = Movies_training.corr()

    # Top 50% Correlation training features with the Value
    top_feature = corr.index[abs(corr['IMDb'] > 0.09)]

    # Correlation plot
    plt.subplots(figsize=(20, 8))
    top_corr = Movies_training[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    multi_reg_start_time = time.time()
    cls1 = linear_model.LinearRegression()
    cls1.fit(X1_train, Y1_train)
    prediction1 = cls1.predict(X1_test)
    multi_reg_end_time = time.time()
    print('multi rag time = ', multi_reg_end_time - multi_reg_start_time)

    print('coef of Multi Linear regression model', cls1.coef_)
    print('Intercept of Multi Linear regression model', cls1.intercept_)
    print('Mean Square Error in Multi Linear regression ', metrics.mean_squared_error(np.asarray(Y1_test), prediction1))
    print('Accuracy of Multi regression', cls1.score(X1_test, Y1_test))
    true_rate_value = np.asarray(Y1_test)[0]
    predicted_rate_value = prediction1[0]
    print('True value for the first movie in the test is : ' + str(true_rate_value))
    print('Predicted value for the first movie in the test set is : ' + str(predicted_rate_value))




