from fastapi import FastAPI
from typing import Optional
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel

app = FastAPI()

from joblib import dump, load

model = load('logistic.joblib')
feature_names = model.feature_names
feature_names.pop()
feature_values = model.feature_values
feature_values.pop()

class ModelInput(BaseModel):
    for x in range(0, len(feature_names)):
        vars()[feature_names[x]] = feature_values[x]
    del x

@app.get("/")
def hello_world():
    return {"Hello World!"}

@app.post("/xab4")
def test_input(modelinput: ModelInput):
    a_dict = vars(modelinput)
    feature_list = []
    for key in a_dict.values():
        feature_list.append(key)
    ytest = model.predict([feature_list])
    return {"prediction": ytest[0]}

