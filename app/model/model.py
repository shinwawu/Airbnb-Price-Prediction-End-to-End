import pandas as pd
import numpy as np 
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
data = 'app/data/raw_data.csv'
path = 'app/model/regressor.pkl'


def normalize_amenities_str(s):
    if pd.isna(s):
        return s
    s2 = str(s).strip()
    # Remove { ... } se presente
    if s2.startswith("{") and s2.endswith("}"):
        s2 = s2[1:-1]
    # Divide por vírgula OU ponto e vírgula
    parts = [p.strip().strip('"') for p in re.split(r'[;,]', s2)]
    # Remove vazios e ordena para ter determinismo
    parts = sorted([p for p in parts if p])
    # Junta com ';' (padrão único)
    return ";".join(parts)

def preprocessing():
    num_cols =[
        'accommodates',
        'bathrooms',
        'latitude',
        'longitude',
        'number_of_reviews',
        'review_scores_rating',
        'bedrooms',
        'beds']
    cat_cols =['property_type',
        'room_type',
        'amenities',
        'bed_type',
        'cancellation_policy',
        'cleaning_fee',
        'city',
        'host_has_profile_pic',
        'host_identity_verified',
        'host_response_rate',
        'instant_bookable']

    num_pip = Pipeline(steps=[("imp",SimpleImputer(strategy='median')),("scale",StandardScaler())])

    cat_pip = Pipeline(steps=[("imp",SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))])

    preprocess = ColumnTransformer(transformers=[("num",num_pip,num_cols),("cat",cat_pip,cat_cols)],remainder='drop')
    pipeline = Pipeline(steps=[("preprocess",preprocess),("model",GradientBoostingRegressor(random_state=42))])
    return pipeline

def train():
    df = pd.read_csv(data)
    y = df["log_price"]
    y_median = df["log_price"].median()
    y = df["log_price"].fillna(y_median)
    df = df.drop(["id","name","log_price","description","first_review","host_since","last_review","neighbourhood",
            "thumbnail_url", "zipcode"],axis=1)
    
    if "amenities" in df.columns:
        df["amenities"] = df["amenities"].apply(normalize_amenities_str)

    x = df
    modelo = preprocessing()
    modelo.fit(x,y)
    save_model(modelo,path)
    
def save_model(model,path):
    joblib.dump(model,path)


def load_model():
    model =joblib.load(path)
    return model

def predict(df):
    modelo = load_model()
    if "amenities" in df.columns:
            df["amenities"] = df["amenities"].apply(normalize_amenities_str)
    ypred=modelo.predict(df)
    return ypred
