import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean(file_path):
    df = pd.read_csv(file_path)
    null_columns = ['zero', 'zero.1', 'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7', 
                    'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13', 'zero.14', 'zero.15', 'zero.16', 'zero.17', 'zero.18']

    df = df.drop(null_columns, axis=1)
    df = df.drop('Passengerid', axis=1)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def feature_engineering(df):
    df['FamilySize'] = df['sibsp'] + df['Parch'] + 1

    def is_alone(size):
        return 1 if size == 1 else 0

    df['IsAlone'] = df['FamilySize'].apply(is_alone)



    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=[1, 2, 3, 4])
    df['FareGroup'] = df['FareGroup'].astype(int)

    def family_group(size):
        if size == 1:
            return 1
        elif size <= 4:
            return 2
        else:
            return 3
        
    df['FamilyGroup'] = df['FamilySize'].apply(family_group)

    def age_group(age):
        if age < 12:
            return 'Child'
        elif age < 18:
            return 'Teenager'
        elif age < 25:
            return 'YoungAdult'
        elif age <55:
            return 'MiddleAge'
        else:
            return 'Elderly'
        
    df['Age'] = df['Age'].apply(age_group)
    df = df.drop('Fare', axis=1)
    df = pd.get_dummies(df, columns=['Age'])
    df = df.drop(['sibsp', 'Parch', 'FamilySize'], axis=1)
    return df


def train():
    df = load_and_clean('train_and_test2.csv')
    df = feature_engineering(df)
    #assigning x and y
    x = df.drop('2urvived', axis=1)
    y = df['2urvived']
    
    #splitting data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    features_median = x_train.median()
    pickle.dump(features_median, open('features_median.pkl', 'wb'))
    #scaling data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    #training model
    model = LogisticRegression(class_weight='balanced')
    model.fit(x_train_scaled, y_train)

    #saving model
    pickle.dump(model,           open('model.pkl', 'wb'))
    pickle.dump(scaler,          open('scaler.pkl', 'wb'))
    pickle.dump(x.columns.tolist(), open('feature_columns.pkl', 'wb'))

    print("Model trained and saved successfully.")
    print(f'features columns: {x.columns.tolist()}')

if __name__ == "__main__":
    train()