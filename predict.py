import pickle
import numpy as np
import pandas as pd

def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    features_median = pickle.load(open('features_median.pkl', 'rb'))
    return model, scaler, feature_columns, features_median

def predict(input_data):
    model, scaler, feature_columns, features_median = load_model('model.pkl')

    data = input_data.copy()
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_columns)
    df = df.fillna(features_median)

    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0] 
    probability = model.predict_proba(scaled)[0, 1]
    return {
        'survived':    int(prediction),
        'probability': round(float(probability), 2),
        'result':      'Survived' if prediction == 1 else 'Did not survive'
    }


