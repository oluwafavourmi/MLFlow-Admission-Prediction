from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.model_selection import train_test_split

with mlflow.start_run():
    # load dataset and train model
    df = pd.read_csv('Admission_Predict.csv')
    x = df[['gre', 'sop', 'cgpa']]
    y = df['admitted']

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    C = 3
    intercept_scaling = 2

    def eval_metrics(actual, pred):
        acc_score = accuracy_score(actual, pred)
        pre_score = precision_score(actual, pred)
        f_score = f1_score(actual, pred)

        return acc_score, pre_score, f_score

    lr = LogisticRegression(C=C, intercept_scaling=intercept_scaling)
    lr= lr.fit(x_train, y_train)

    prediction = lr.predict(x_test)

    #log metrics
    (acc_score, pre_score, f_score) = eval_metrics(y_test, prediction)
    print('Logistic Regression (C={:f}, intercept_scaling{:f}:'.format(C, intercept_scaling))
    print("accuracy score: %s" % acc_score)
    print("precision score score: %s" % pre_score)
    print("f1 score: %s" % f_score)

    mlflow.log_metric('accuracy score', acc_score)
    mlflow.log_metric('precision score', pre_score)
    mlflow.log_metric('f1 score', f_score)

    # log model params
    mlflow.log_param("C", lr.C)
    mlflow.log_param("Intercept Scaling", lr.intercept_scaling)

    #for remote server
    remote_server_uri = 'https://dagshub.com/oluwafavourmi/MLFlow-Admission-Prediction.mlflow'
    mlflow.set_tracking_uri(remote_server_uri)
    trackong_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model Registry does not work with file store
    if trackong_url_type_store != "file":
        mlflow.sklearn.log_model(lr, "model", registered_model_name='Log-regression-admission')
    else:
        # log model
        mlflow.sklearn.log_model(lr, "model")