import multiple_linear_regression2 as ML_R
import linear_regression as L_R
from pickle import load
import poly_reg as P_L
import SVC as svc
import pandas as pd
from pre_processing import *
from pre_processing_with_norm import *
from classifiction import *

def main():

    # read data and make pre-processing#
    #data = pd.read_csv('Movies_training.csv')
    #c_data = pd.read_csv('Movies_training_classification.csv')
    #data=pd.read_csv('Movies_testing.csv')
    c_data = pd.read_csv('Movies_testing_classification.csv')

    Movies_training=process(c_data,1)
    #Movies_training = process_with_norm(c_data, 1)
    #Movies_training = process(data,0)
    #Movies_training = process_with_norm(data,0)

    # Load data and make pre-processing
    c_Movies_training = load(open('c_t_data.pkl', 'rb'))
    #c_Movies_training = load(open('c_data.pkl', 'rb'))
    #c_Movies_training = load(open('c_data_N.pkl', 'rb'))
    #Movies_training = load(open('T_data.pkl', 'rb'))
    #Movies_training = load(open('data.pkl', 'rb'))
    #Movies_training = load(open('data_N.pkl', 'rb'))
    #print(Movies_training.to_string())

    #classifiction(c_Movies_training)
    #ML_R.Mult_reg(Movies_training)
    #P_L.polynomial_reg(Movies_training)
    #L_R.lin_reg(Movies_training)
    #svc.SVC_reg(Movies_training)

if __name__ == '__main__':
    main()