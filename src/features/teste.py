from features.feature_store import FeatureStore
import pandas as pd

sample = {
    "person_income": 9900,
    "person_home_ownership": "OWN",
    "person_emp_length": 2.0,
    "loan_intent": "VENTURE",
    "loan_grade": "A",
    "loan_amnt": 2500,
    "loan_int_rate": 7.14,
    "loan_status": 1,
    "loan_percent_income": 0.25,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 2,
    "faixa_etaria": "20-29"
}

df = pd.DataFrame([sample])
df = df.drop(columns=["loan_status"], errors="ignore")

feature_store = FeatureStore.load()
X = feature_store.transform(df)
print(X)
