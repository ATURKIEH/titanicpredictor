import pickle
import numpy as np
import pandas as pd

def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    return model, scaler, feature_columns

def predict(input_data):
    model, scaler, feature_columns = load_model('model.pkl')
    
    data = input_data.copy()
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0] 
    probability = model.predict_proba(scaled)[0, 1]
    return {
        'survived':    int(prediction),
        'probability': round(float(probability), 2),
        'result':      'Survived' if prediction == 1 else 'Did not survive'
    }

if __name__ == "__main__":
    sample = {
    'Sex':            1,
    'Pclass':         1,
    'Embarked':       0,  
    'IsAlone':        1,
    'FareGroup':      4,
    'FamilyGroup':    1,
    'Age_Child':      0,
    'Age_Teenager':   0,
    'Age_YoungAdult': 1,
    'Age_MiddleAge':  0,
    'Age_Elderly':    0,
}

    result = predict(sample)
    print(f"Survived:    {result['survived']}")
    print(f"Probability: {result['probability']}")
    print(f"Result:      {result['result']}")
