import uvicorn
from fastapi import FastAPI
import pickle
import sys
import json
from sklearn.linear_model import LogisticRegression


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


# Init app
print(__name__)
app = FastAPI()
print("Hello")

# Routes
@app.get('/')
async def index():
    return {"text":"Hello API Masters"}


@app.get('/items/{name}')
async def get_items(name):
    return {"name":name}


@app.get('/predict/')
async def predict():
    with open('customer.json') as json_file:
        customer = json.load(json_file)

    model_file = 'model_C=1.0.bin'

    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    pred = predict_single(customer, dv, model)
    return {"Prediction":pred}


if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)