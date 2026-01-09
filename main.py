from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional,List
from app.model.model import train,load_model
app= FastAPI(title='Airbnb Price Prediction')
model = load_model()
class reqpredict(BaseModel):
    accommodates : Optional[List[Optional[float]]] = None
    bathrooms : Optional[List[Optional[float]]] = None
    latitude : Optional[List[Optional[float]]] = None
    longitude: Optional[List[Optional[float]]] = None
    number_of_reviews: Optional[List[Optional[float]]] = None
    review_scores_rating: Optional[List[Optional[float]]] = None
    bedrooms: Optional[List[Optional[float]]] = None
    beds: Optional[List[Optional[float]]] = None
    property_type : Optional[List[Optional[str]]] = None
    room_type : Optional[List[Optional[str]]] = None
    amenities: Optional[List[Optional[str]]] = None
    bed_type: Optional[List[Optional[str]]] = None
    cancellation_policy: Optional[List[Optional[str]]] = None
    cleaning_fee: Optional[List[Optional[str]]] = None
    city: Optional[List[Optional[str]]] = None
    host_has_profile_pic: Optional[List[Optional[str]]] = None
    host_identity_verified: Optional[List[Optional[str]]] = None
    host_response_rate: Optional[List[Optional[str]]] = None
    instant_bookabl: Optional[List[Optional[str]]] = None

@app.health('/health')
def health():
    return {'status' : 'ok'}

def predict(req:reqpredict):
    global model
    if model == None:
        train()
        model = load_model()
    df = pd.DataFrame({"accommodates" : req.accommodates, "bathrooms" : req.bathrooms, "latitude" : req.latitude,
                       "longitude" : req.longitude,'number_of_reviews' : req.number_of_reviews, "review_scores_rating" : req.review_scores_rating,
                       'bedrooms' : req.bedrooms, 'beds' : req.beds, 'property_type' : req.property_type, 'room_type' : req.room_type,
                       'amenities' : req.amenities,'bed_type' : req.bed_type,'cancellation_policy' : req.cancellation_policy,
                       'cleaning_fee' : req.cleaning_fee, 'city' : req.city,'host_has_profile_pic' : req.host_has_profile_pic,
                       'host_identity_verified' : req.host_identity_verified, 'host_response_rate' : req.host_response_rate,
                       'instant_bookabl' : req.instant_bookabl
                       })
    ypred = model.predict(df)
    return {"predictions" : ypred.tolist()}
