from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

from app.model_loader import load_model


app = FastAPI(title='Iris classifier api')

model = load_model('83cdc2c12712403389f809ce869b64f8')

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict_iris(data:IrisInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = model.predict(input_data)[0]
    return {'predicted class':int(pred)}