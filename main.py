from fastapi import FastAPI
from models import House
import sklearn
app = FastAPI()
import pickle
import numpy as np
@app.get("/")
def home():
    print("Welcome to Home page")
    return {"Message": "Welcome !!!!"}


with open("model.pkl", "rb") as file:
    trained_model = pickle.load(file)

@app.post("/predict")
def predict(data : House):
    x = np.array(data.area).reshape(-1,1)
    prediction = trained_model.predict(x)
    result = prediction.flatten()
    print(result)
    return {"Cost": p for p in result}
