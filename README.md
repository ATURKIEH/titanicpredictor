# Titanic Survival Classifier
 
Predicts passenger survival on the Titanic using classification models. Focuses on feature engineering and class imbalance handling to improve recall on the minority class (survivors).
 
---
 
## Results
 
| Model | Accuracy | Recall | F1 |
|-------|----------|--------|----|
| **Logistic Regression** | **0.736** ✅ | **0.753** | **0.614** |
| Random Forest (tuned) | 0.740 | 0.739 | 0.613 |
| XGBoost | 0.732 | 0.726 | 0.602 |
 
**Key finding:** Logistic Regression achieved the best F1 and recall after applying `class_weight='balanced'` to handle the 73/27 class imbalance. Recall was prioritized over accuracy — missing a survivor is worse than a false positive.
 
---
 
## Dataset
 
Custom Titanic dataset — 1,309 samples. Place `train_and_test2.csv` in the project root.
 
---
 
## Feature Engineering
 
Raw features alone were insufficient. The following engineered features improved model performance:
 
- **FamilySize** — SibSp + Parch + 1
- **IsAlone** — binary flag for solo travelers (lower survival rate)
- **FamilyGroup** — solo (1), small family 2-4 (2), large family 5+ (3)
- **FareGroup** — fare quartiles 1-4 via pd.qcut
- **AgeGroup** — Child (<12), Teenager (12-18), YoungAdult (18-25), MiddleAge (25-55), Elderly (55+) — one-hot encoded
 
---
 
## Project Structure
 
```
├── train.py       # Data pipeline + model training → saves pkl files
├── predict.py     # Loads saved model → returns survival prediction
├── app.py         # FastAPI REST API — exposes predict.py via HTTP
├── main.py        # Full evaluation, model comparison, visualizations
├── requirements.txt
└── .gitignore
```
 
---
 
## Setup
 
```bash
pip install -r requirements.txt
python train.py       # generates pkl files
python predict.py     # runs a sample prediction
python main.py        # full evaluation and plots
```
 
---
 
## API Usage
 
```bash
uvicorn app:app --reload
```
 
Then open `http://localhost:8000/docs` for the interactive API UI.
 
**POST** `/predict`
 
```json
{
  "features": {
    "Sex": 1,
    "Pclass": 1,
    "Embarked": 0,
    "IsAlone": 1,
    "FareGroup": 4,
    "FamilyGroup": 1,
    "Age_Child": 0,
    "Age_Teenager": 0,
    "Age_YoungAdult": 0,
    "Age_MiddleAge": 1,
    "Age_Elderly": 0
  }
}
```
 
Sex: 1=female, 0=male | Embarked: 0=S, 1=C, 2=Q
 
**Response:**
```json
{
  "survived": 1,
  "probability": 0.89,
  "result": "Survived"
}
```
 
---
 
## Tech Stack
 
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, FastAPI, Uvicorn
 
---
## Running With Docker

```bash
docker pull aturkieh/titanic-classifier
docker run -p 8000:8000 aturkieh/titanic-classifier
```

Then visit http://localhost:8000/docs