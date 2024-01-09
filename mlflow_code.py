from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from mlflow.pyfunc import PyFuncModel
from sklearn.linear_model import LogisticRegression
import tkinter.filedialog as filedialog
import tkinter as tk

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.types.schema import Schema


def data_upload():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    data = pd.read_csv(file_path)
    return data


def preprocess(data):
    dataset = data.drop(columns=['STOP', 'PATIENT', 'ENCOUNTER', 'START'], axis=1)

    dataset.dropna(how='any', inplace=True)

    # label_encoding target column
    label_encoder = preprocessing.LabelEncoder()

    dataset['SEVERITY2'] = label_encoder.fit_transform(dataset['SEVERITY2'])
    y = dataset['SEVERITY2']
    dataset.pop('SEVERITY2')

    # One-hot encoding
    encoded_data = pd.get_dummies(dataset)

    return encoded_data, y


def feature_eng(encoded_data, y):
    # feature selection
    chi_values = chi2(encoded_data, y)
    chi_value2 = pd.Series(chi_values[1])
    chi_value2.index = encoded_data.columns
    chi_value3 = chi_value2.sort_index(ascending=False)

    X = encoded_data[['TYPE_intolerance', 'TYPE_allergy', 'SYSTEM_SNOMED-CT', 'SYSTEM_RxNorm', 'SEVERITY1_SEVERE',
                      'SEVERITY1_MODERATE', 'SEVERITY1_MILD']].copy()
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test


def eval_metrics(actual, pred):
    f1 = f1_score(actual, pred, average='micro')
    roc = roc_auc_score(actual, pred)
    cm = confusion_matrix(actual, pred)
    return f1, roc, cm


def parameter():
    regular = input('''Enter regularization parameter"
              Hint: try lower values like 1,2,3 for better results''')
    if (regular == '' or regular != int):
        regular = 1
    p = input('''Enter penalty parameter
        Hint: l1, l2, elasticnet, none''')
    if p == '':
        p = 'l2'

    s = input('''Enter solver
        Hint: newton-cg, lbfgs, liblinear, sag, saga''')
    if s == '':
        s = 'lbfgs'
    return regular, p, s


if __name__ == "__main__":

    data = data_upload()
    encoded_data, y = preprocess(data)
    X_train, X_test, y_train, y_test = feature_eng(encoded_data, y)

    # model training using MLflow
    regular, penalty, solver = parameter()

    client = MlflowClient()
    experiment_name = "Experiment Tracking"
    try:
        # Create experiment
        experiment_id = client.create_experiment(experiment_name)
    except:
        # Get the experiment id if it already exists
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    remote_uri = ""
    mlflow.set_tracking_uri(remote_uri)

    with mlflow.start_run(experiment_id='1') as run:
        # run_id = run.info.run_uuid
        '''

        MlflowClient().set_tag(run_id,
                               "mlflow.note.content",
                               "An end-to-end implementation of ML model using synthea-allergies data")'''

        # Define custom tag
        tags = {"Application": "ML flow implementation",
                "release.version": "2.2.0"}
        # Set Tag
        mlflow.set_tags(tags)
        # mlflow.sklearn.autolog()

        # train model

        log_reg = LogisticRegression(penalty=penalty, solver=solver, C=regular)
        log_reg.fit(X_train, y_train)

        # perform prediction on test data
        y_pred = log_reg.predict(X_test)
        (f1, roc, cm) = eval_metrics(y_test, y_pred)
        t_p, t_n, f_p, f_n = cm.ravel()

        # log metrics and parameters
        mlflow.log_metric("t_p", t_p)
        mlflow.log_metric("t_n", t_n)
        mlflow.log_metric("f_p", f_p)
        mlflow.log_metric("f_n", f_n)

        mlflow.log_param("solver", solver)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("c", regular)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc", roc)
        mlflow.sklearn.log_model(log_reg, "model")

        # log model

        signature = infer_signature(X_train, y_pred)
        input_example = {
            "TYPE_intolerance": 1,
            "TYPE_allergy": 1,
            "SYSTEM_SNOMED-CT": 0,
            "SYSTEM_RxNorm": 1,
            "SEVERITY1_SEVERE": 0,
            "SEVERITY1_MODERATE": 1,
            "SEVERITY1_MILD": 0
        }
        input_schema = Schema([
            mlflow.types.ColSpec("integer", "TYPE_intolerance"),
            mlflow.types.ColSpec("integer", "TYPE_allergy"),
            mlflow.types.ColSpec("integer", "SYSTEM_SNOMED-CT"),
            mlflow.types.ColSpec("integer", "SYSTEM_RxNorm"),
            mlflow.types.ColSpec("integer", "SEVERITY1_SEVERE"),
            mlflow.types.ColSpec("integer", "SEVERITY1_MODERATE"),
            mlflow.types.ColSpec("integer", "SEVERITY1_MILD")

        ])
        mlflow.sklearn.log_model(
            sk_model=log_reg,
            artifact_path="model",
            registered_model_name="logistic_reg", signature=signature, input_example=input_example,

        )
