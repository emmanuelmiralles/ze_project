from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import ast
import copy

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.get("/")
def index():
    return {"desole": "pas de loyer"}
@app.get("/predict")
def predict(surfa, surfb):
    model=joblib.load('../model.joblib')
    X = pd.DataFrame(
        {
            'GrLivArea':int(surfa),
            'RoofSurface':int(surfb)
        },index=[0])
    y_pred=model.predict(X)
    return {'prediction':y_pred[0]}