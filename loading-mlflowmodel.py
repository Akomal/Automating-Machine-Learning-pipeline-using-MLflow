
import pandas as pd
from mlflow.pyfunc import PyFuncModel
import mlflow




#Retrieving results and making predictions on new data using registered model
logged_model =""#runid against which you want to get predictions

    # Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
import numpy as np
data = pd.DataFrame(
{"TYPE_intolerance": [1], "TYPE_allergy": [0], "SYSTEM_SNOMED-CT": [0], "SYSTEM_RxNorm": [0],
         "SEVERITY1_SEVERE": [1], "SEVERITY1_MODERATE": [1], "SEVERITY1_MILD": [1]})
data["TYPE_intolerance"] = data["TYPE_intolerance"].fillna(0).astype(np.int32, errors='ignore')
data["TYPE_allergy"] = data["TYPE_allergy"].fillna(0).astype(np.int32, errors='ignore')
data["SYSTEM_SNOMED-CT"] = data["SYSTEM_SNOMED-CT"].fillna(0).astype(np.int32, errors='ignore')
data["SYSTEM_RxNorm"] = data["SYSTEM_RxNorm"].fillna(0).astype(np.int32, errors='ignore')
data["SEVERITY1_SEVERE"] = data["SEVERITY1_SEVERE"].fillna(0).astype(np.int32, errors='ignore')
data["SEVERITY1_MODERATE"] = data["SEVERITY1_MODERATE"].fillna(0).astype(np.int32, errors='ignore')
data["SEVERITY1_MILD"] = data["SEVERITY1_MILD"].fillna(0).astype(np.int32, errors='ignore')

print(loaded_model.predict(data))