import joblib
import os
import pandas as pd
from pathlib import Path
import warnings

# WARNING: This suppresses warning because sklearn dumped model is from version 0.22
#          and the current sklearn version is 0.24
warnings.simplefilter('ignore')

PROJECT_DIR = Path(__file__).parents[2]
COLUMN_ORDER = ['cylinder', 'displacement', 'horsepower', 'weight', 
                'acceleration','model_year', 'origin']

mlmodel_path = os.path.join(PROJECT_DIR, 'models', 'extratrees.pkl')
preprocessor_path = os.path.join(PROJECT_DIR, 'models', 'preprocessor.pkl')

print(mlmodel_path)

mlmodel = joblib.load(mlmodel_path)
preprocessor = joblib.load(preprocessor_path)


def predict_mpg(config, mlmodel=mlmodel, preprocessor=preprocessor):
    if isinstance(config, (list, dict)):
        df = pd.DataFrame(config)
    else:
        df = config

    df = df[COLUMN_ORDER]

    df_prep = preprocessor.transform(df)
    preds = mlmodel.predict(df_prep)

    return preds


if __name__ == '__main__':
    vehicle_config = {'acceleration': [17.0],
                        'cylinder': [6],
                        'displacement': [250.0],
                        'horsepower': [100.0],
                        'model_year': [74],
                        'origin': [1],
                        'weight': [3336.0]}

    print(predict_mpg(vehicle_config, mlmodel, preprocessor))