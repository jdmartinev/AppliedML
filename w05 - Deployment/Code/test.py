import pickle
import sys
import json
from sklearn.linear_model import LogisticRegression


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

with open(sys.argv[2]) as json_file:
    customer = json.load(json_file)

model_file = sys.argv[1]

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
pred = predict_single(customer, dv, model)
    
print(pred)