import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


#cleanign data
df = pd.read_csv('train_and_test2.csv')
null_columns = ['zero', 'zero.1', 'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7', 
                'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13', 'zero.14', 'zero.15', 'zero.16', 'zero.17', 'zero.18']

df = df.drop(null_columns, axis=1)
df = df.drop('Passengerid', axis=1)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#adding features
df['FamilySize'] = df['sibsp'] + df['Parch'] + 1

def is_alone(size):
    return 1 if size == 1 else 0

df['IsAlone'] = df['FamilySize'].apply(is_alone)



df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=[1, 2, 3, 4])

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


#assigning x and y
x = df.drop('2urvived', axis=1)
y = df['2urvived']

# splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#fit and train

#---------Logistic Regression-----------------
model = LogisticRegression(class_weight='balanced')
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Logistic Regression Results:")
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{cm}')
print(df.columns.tolist())


#---------Rforest Classifier-----------------


params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    params, cv=5, scoring='f1', n_jobs=-1
)
grid.fit(x_train_scaled, y_train)
y_pred_rf = grid.predict(x_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
cfm = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Classifier Results:")
print(grid.best_params_)
print(grid.best_score_)

print(f'Accuracy: {accuracy_rf}')
print(f'Precision: {precision_rf}')
print(f'Recall: {recall_rf}')
print(f'F1 Score: {f1_rf}')
print(f"Confusion Matrix:")
print(cfm)



#---------XGBoost Classifier-----------------
xgb_model = XGBClassifier(scale_pos_weight=2.7,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42)

xgb_model.fit(x_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(x_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nXGBoost Classifier Results:")
print(f'Accuracy: {accuracy_xgb}')
print(f'Precision: {precision_xgb}')
print(f'Recall: {recall_xgb}')
print(f'F1 Score: {f1_xgb}')
print(f'Confusion Matrix:\n{cm_xgb}')

#cross validation
for name, clf in [
    ('Logistic Regression', model),
    ('Random Forest', grid.best_estimator_),
    ('XGBoost', xgb_model)
]:
    scores = cross_val_score(clf, x, y, cv=5, scoring='f1')
    print(f"{name} CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")





